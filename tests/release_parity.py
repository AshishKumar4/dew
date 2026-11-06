"""A released decision model's parity with its own engine, as a declaration (row 4485): the release, the
requests its engine answered and the recording of its answers (`tools/recorded_references.record_case`), which
one harness lays out through Dew's public loader and judges by check 1 (`assert_answers_as_its_engine`).

A recording's site holds, stacked on its seeds, the rows the engine laid (`tokens` `[R, L]`, `lengths` `[R]`,
`options` `[R, Q, O]` or `[R, O]` the real options, `spans` their spans, `blocks` where its rows are read in
blocks) and its answers (`scores`, each real option's logit; a `probe` beside them, the gradients of their
probed sum the engine took, `gradient.hidden` and `gradient.<name>` of the head's parameters). A release is
declared by its directory, its requests, the site, and the names its engine gives the head's parameters.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from reference_error import as_given, assert_as_exact, declared_seeds, indexed_recording, recording

from dew.batch import part


@dataclass(frozen=True)
class ReleaseParity:
    """One release's parity with its engine at one recorded site."""

    name: str
    release: Callable[[Path], Path]
    """The release's directory, prepared under a scratch directory where it is written for this site."""
    recording: Path
    site: str
    requests: Path
    """The wire requests of each seed, `[[{state, questions: {id: wire}}, ...], ...]`."""
    gradients: Mapping[str, str] = field(default_factory=dict)
    """Each recorded `gradient.<name>`'s head parameter, by its name in the head's tree."""


def assert_answers_as_its_engine(parity: ReleaseParity, scratch: Path) -> None:
    """Load `parity`'s release through `Decide.from_pretrained`, lay each seed's requests out, assert the
    rows the engine laid exactly (tokens, option spans, blocks), and judge its scores and the gradients the
    recording holds by check 1."""
    from dew.decision import Decide, DecisionInputs, Question

    task = Decide.from_pretrained(parity.release(scratch), dtype="float32", attention_impl="reference")
    seeds = declared_seeds(parity.recording)[parity.site]
    draw, reference, truth = indexed_recording(recording(parity.recording), parity.site, seeds)
    requests = json.loads(parity.requests.read_text())
    head = task.model.head

    @jax.jit
    def forward(inputs, probe):
        states = task.model.states(task.variables, inputs)
        table = None if head.draws_from_rows else task.model.output_table(task.variables)

        def observed(params, hidden):
            ids = head.tied_labels()
            tied = None if ids is None else task.model.readout_logits(task.variables, hidden, inputs, ids)
            scores = head.apply({"params": params}, hidden, inputs, table, tied=tied)[:, 0]
            return jnp.sum(scores * probe)

        gradient, hidden = jax.grad(observed, (0, 1))(part(task.variables, "head").get("params", {}), states)
        return {"scores": task.model.logits(task.variables, inputs)[:, 0], "gradient.hidden": hidden,
                **{f"gradient.{name}": gradient[native] for name, native in parity.gradients.items()}}

    @as_given
    def dew(index, held):
        rows = []
        for request in requests[int(index)]:
            questions = {name: Question.from_wire(wire) for name, wire in request["questions"].items()}
            rows.extend(task.layout.rows(task.tokenizer, task.specials, request["state"], questions))
        laid = DecisionInputs.collate(rows, task.specials.pad, options=held["options"].shape[1])
        mask = np.arange(held["tokens"].shape[1])[None] < held["lengths"][:, None]
        for row, encoded in enumerate(rows):
            np.testing.assert_array_equal(encoded.tokens, held["tokens"][row, :held["lengths"][row]])
            np.testing.assert_array_equal(encoded.questions[0].options,
                                          held["spans"][row, held["options"][row]])
            if "blocks" in held:
                np.testing.assert_array_equal(encoded.blocks, held["blocks"][row, :held["lengths"][row]])
        inputs = replace(laid, tokens=jnp.asarray(held["tokens"]), segments=jnp.asarray(mask, jnp.int32),
                         positions=jnp.broadcast_to(jnp.arange(mask.shape[1]), mask.shape),
                         blocks=jnp.asarray(held["blocks"]) if "blocks" in held else None)
        probe = np.zeros(held["options"].shape, np.float32)
        probe[held["options"]] = held["probe"]
        result = forward(inputs, jnp.asarray(probe))
        assert np.isneginf(np.asarray(result["scores"])[~held["options"]]).all()
        return {**result, "scores": np.asarray(result["scores"])[held["options"]]}

    assert_as_exact(dew, reference, draw, f"{parity.name} as its engine answers", seeds=seeds,
                    truth_fn=truth)
