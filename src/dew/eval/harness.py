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

import dataclasses
import functools
from collections.abc import Sequence
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import handle_stop_sequences, normalize_gen_kwargs, postprocess_generated_text

from dew.batch import BLOCKS_FIELD
from dew.distributed.placement import Layout, MeshSpec
from dew.inference.tasks import TextGeneration, cache_ceiling
from dew.interop.generation_config import source_decoding
from dew.nn.inputs import ModelInputs, pad_token_rows
from dew.nn.kernels.options import program
from dew.nn.visibility import served_blocks
from dew.objectives.lm.objective import LMObjective, TokenScorer

DEFAULT_CONTEXT = 2048
"""The scoring window for a model that declares no `max_seq_len`."""

DEFAULT_BUDGET = 256
"""The generation budget of a request that names none, `HFLM.max_gen_toks`."""

TokenRequest = tuple[tuple[str, str] | None, list[int], list[int]]
"""One `_loglikelihood_tokens` request: the strings, context ids, continuation ids."""


def _scored(scorer: TokenScorer, variables, rows: ModelInputs) -> tuple[jax.Array, jax.Array]:
    """Return the per-target log-probability and whether the target was the argmax, as the scorer
    the language-model losses share scores them (`TokenScorer.token_scores`), through its tiled
    head, which holds a tile of the vocabulary at a time rather than every row's whole logits;
    compiled with the model's XLA options (`program`).

    `rows` is `[B, T]` of token ids, with their fields (`_scoring_inputs`); both results are
    `[B, T - 1]`, entry `i` belonging to `rows[:, i + 1]`. Rows are right-padded by the caller, which
    a causal model cannot read backwards, so a short row's own slots hold what they would hold alone.
    """
    scores = scorer.token_scores(variables, rows, predict=True)
    return -scores.losses, scores.correct == 1


def _scoring_inputs(rows: list[list[int]], contexts: list[int],
                    prefix: int | Literal["prompt"] | None) -> ModelInputs:
    """`rows`, each its context's first `contexts` ids and its continuation after, right-padded, as the
    scorer reads them: under a prefix run, each context's blocks a server makes of a prompt
    (`served_blocks`, the context whole under "prompt"), each continuation token and padding slot in
    no block (-1), and the padding marked, so a continuation is scored as its generation reads it."""
    tokens, fields = pad_token_rows(rows, padding_side="right")
    token_fields = {name: jnp.asarray(value) for name, value in fields.items()}
    if prefix is not None:
        context = np.arange(tokens.shape[1])[None, :] < np.asarray(contexts)[:, None]
        blocks = np.where(context, served_blocks(context, prefix), -1).astype(np.int32)
        token_fields[BLOCKS_FIELD] = jnp.asarray(blocks)
    return ModelInputs(jnp.asarray(tokens), token_fields=token_fields)


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
        self._scorers: dict[int, TokenScorer] = {}

    @classmethod
    def from_run(cls, run: str, *, batch_size: int = 1, ema: bool | None = None,
                 step: int | str | None = None, mesh: MeshSpec | Mesh | None = None,
                 layout: Layout | None = None, dtype: str | None = None, param_dtype: str | None = None,
                 trust: Sequence[str] = ()) -> DewLM:
        """Load the run directory `run` as a harness model, the way `dew.pipeline` builds it.

        With `ema=None`, it uses the run's averaged weights when the run kept
        them, and its live weights otherwise. `mesh`, `layout`, `dtype`, `param_dtype` and
        `trust` are as `TextGeneration.from_run` takes them: a run sharded over a mesh, its
        weights stored narrower. `--model_args` names the mesh by its axes' sizes,
        `run=<directory>,fsdp=4,param_dtype=bfloat16` (`MeshSpec`'s fields).
        """
        return cls(TextGeneration.from_run(run, ema=ema, step=step, mesh=mesh, layout=layout, dtype=dtype,
                                           param_dtype=param_dtype, trust=trust),
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
        device = arguments.pop("device", None)
        if device is not None:
            raise ValueError(
                f"--device {device} names a torch device, and a run's weights are placed on JAX "
                f"devices when its task is built; leave --device unset")
        run = arguments.pop("run", None)
        if run is None:
            raise ValueError(
                "--model dew evaluates a run directory; name it with "
                "--model_args run=<directory>")
        batch = arguments.pop("batch_size", 1)
        # A mesh's fields are each a key of their own, as one arguments string cannot hold a record.
        mesh = {field.name: arguments.pop(field.name) for field in dataclasses.fields(MeshSpec)
                if field.name in arguments}
        return cls.from_run(str(run), batch_size=int(batch), mesh=MeshSpec(**mesh) if mesh else None,
                            **dict(arguments))

    @property
    def eot_token_id(self) -> int:
        """The id a row with no context is conditioned on: the sampling policy's first EOS. A task
        whose policy names none has no such id, and is refused (ValueError) rather than given one."""
        stops = self.task.sampling.stops
        if not stops:
            raise ValueError("the task's sampling policy names no EOS id, the end-of-text id a row with "
                             "no context is conditioned on; give the task's sampling eos_token_ids")
        return stops[0]

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
                   left_truncate_len: int | None = None, **kwargs: int | None) -> list[int]:
        """Encode `string` with the run's own tokenizer into one row of ids, as `HFLM.tok_encode` does.

        `add_special_tokens=True` is the tokenizer's own encoding, its special tokens around the text
        (a BOS before it, T5's EOS after it); False is the text's ids alone. None, the default every
        scored context and continuation reads, opens the text as the tokenizer opens a sequence and
        closes it with nothing (`_opening`): a context and its continuation encode as one row, so an
        EOS a tokenizer closes a sequence with would end the context before its continuation,
        shifting where the continuation starts and scoring an EOS it never wrote. The opening is
        left off where the text already starts with its spelling, as `HFLM` leaves its BOS off where
        the text starts with the BOS's decoded text (`Processor.spelling`), so no row starts with two.
        `left_truncate_len` keeps that many of the last ids. Any other option raises TypeError, since
        `HFLM` reads none.
        """
        if kwargs:
            raise TypeError(f"tok_encode takes no {', '.join(sorted(kwargs))}")
        if add_special_tokens:
            ids = self._processor.encode(string, specials=True)
        else:
            ids = self._processor.encode(string, specials=False)
            if add_special_tokens is None and not string.startswith(self._opened):
                ids = [*self._opening, *ids]
        return ids[-left_truncate_len:] if left_truncate_len else ids

    @functools.cached_property
    def _opened(self) -> str:
        """The text the run's tokenizer spells its opening with (`_opening`, `Processor.spelling`)."""
        return self._processor.spelling(self._opening)

    @functools.cached_property
    def _opening(self) -> list[int]:
        """The special ids the run's tokenizer opens a sequence with: those its own encoding of a
        probe holds before the probe's ids (Llama's BOS; none for GPT-2's, which declares a BOS and
        inserts none, or T5's, whose one special closes a sequence). The probe is one letter, since a
        tokenizer may refuse an empty row."""
        whole = self._processor.encode("a", specials=True)
        bare = self._processor.encode("a", specials=False)
        for start in range(len(whole) - len(bare) + 1):
            if whole[start:start + len(bare)] == bare:
                return whole[:start]
        raise ValueError("the run's tokenizer encodes a text with its special tokens around ids other "
                         "than the text's own")

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
        greedy, and win every multiple-choice comparison it is in. Any
        option besides `disable_tqdm` raises TypeError.
        """
        del disable_tqdm
        if kwargs:
            raise TypeError(f"_loglikelihood_tokens takes no {', '.join(sorted(kwargs))}")
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
            inputs = _scoring_inputs([rows[index] for index in batch],
                                     [len(rows[index]) - len(requests[index][2]) for index in batch],
                                     self.task.prefix)
            scoring = program(_scored, self.task.model, static_argnums=(0,))
            probabilities, argmax = scoring(self._scorer(inputs.tokens.shape[1]), self.task.variables, inputs)
            values, matched = np.asarray(probabilities), np.asarray(argmax)
            for offset, index in enumerate(batch):
                end = len(rows[index]) - 1
                start = end - len(requests[index][2])
                answers.append((float(values[offset, start:end].sum()),
                                bool(matched[offset, start:end].all())))
        return answers

    def _scorer(self, width: int) -> TokenScorer:
        """The task's model scored over rows of `width` ids as a language model's loss scores them
        (`LMObjective`), every target counted and its head tiled (`_scored`), one for each width a
        batch pads to."""
        if width not in self._scorers:
            self._scorers[width] = LMObjective(self.task.model, width - 1, head_tile="tiled")
        return self._scorers[width]

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

        A beam search (`num_beams` above 1) raises ValueError. `HFLM` always
        stops on its EOS's text as well as its stop strings, and its criterion
        reads the leading candidate alone and ends every beam together, which
        neither Dew's per-beam search nor a cut of its answer reproduces.
        A request's `max_length` is the total length `HFLM` hands `generate`,
        context included, in place of the budget. Its temperature is 0 where
        it names none, which decodes greedily unless the request asks to
        sample; sampling at 0 raises ValueError, as `generate` does. The other
        controls are the ones `HFLM` hands Transformers' `generate`, and they
        decode as a source's generation config does
        (`dew.interop.generation_config.source_decoding`), over the task's own
        EOS and pad ids where the request names none: sampling filters,
        penalties and logits processors. One Dew has no counterpart for raises
        ValueError, as `generate` refuses a keyword it does not use.
        """
        del disable_tqdm
        eos = None if self.task.sampling.eos_token_ids in (None, ()) else self.tok_decode([self.eot_token_id])
        answers = []
        for request in requests:
            arguments = list(request.args)
            context, raw = str(arguments[0]), dict(arguments[1])
            # normalize_gen_kwargs sets the budget and the stop strings, and
            # makes do_sample=False greedy; the type leaves every key optional.
            controls: dict[str, Any] = dict(normalize_gen_kwargs(raw, DEFAULT_BUDGET))
            budget = controls.pop("max_gen_toks", DEFAULT_BUDGET)
            stops = handle_stop_sequences(controls.pop("until", None), eos=eos)
            if int(controls.get("num_beams") or 1) > 1:
                raise ValueError(
                    "a beam search is refused: lm-eval's own model stops on its EOS's text and stop "
                    "strings by the leading candidate alone and ends every beam together, which "
                    "Dew's per-beam search would answer differently")
            seed = int(controls.pop("seed", 0))
            total = controls.pop("max_length", None)
            # HFLM._model_generate: no temperature is 0, which is greedy unless sampling is asked for.
            controls.setdefault("temperature", 0.0)
            if controls["temperature"] == 0.0 and controls.get("do_sample") is None:
                controls["do_sample"] = False
            if controls.get("do_sample") and controls["temperature"] == 0.0:
                raise ValueError("sampling at temperature 0 is refused, as Transformers' generate refuses it")
            task_ids = {"eos_token_id": self.task.sampling.eos_token_ids,
                        "pad_token_id": self.task.sampling.pad_token_id}
            sampling, chain, strategy = source_decoding(task_ids, controls, self.task.model, 1, None)
            if budget >= self.max_length:
                raise ValueError(
                    f"a budget of {budget} ids leaves no room for a context in this model's "
                    f"{self.max_length}-id window; lm-eval's HFLM refuses it too")
            ids = self.tok_encode(context)[-(self.max_length - budget):]
            drawn_budget = budget if total is None else int(total) - len(ids)
            text = "" if drawn_budget < 1 else self.task.decode(self.task(
                [ids], drawn_budget, key=seed, sampling=sampling, logits=chain, strategy=strategy))[0]
            answers.append(postprocess_generated_text(text, stops, None))
        return answers


__all__ = ["DewLM"]
