"""A released decision model's parity with its own engine, as a declaration (row 4485): the release, the
requests its engine answered and the recording of its answers (`tools/recorded_references.record_case`), which
one harness lays out through Dew's public loader and judges by check 1 (`assert_answers_as_its_engine`).

A recording's site holds, stacked on its seeds, the rows the engine laid (`tokens` `[R, L]`, `lengths` `[R]`,
`options` `[R, Q, O]` or `[R, O]` the real options, `spans` their spans, `blocks` where its rows are read in
blocks, and where its requests show pictures, `images` `[R, P, H, W, 3]` and how many of them each request
carries, `image_counts` `[R]`) and its answers (`scores`, each real option's logit; a `probe` beside them,
the gradients of their probed sum the engine took, `gradient.hidden` and `gradient.<name>` of the head's
parameters). A release is
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
from dew.nn.inputs import ModelInputs


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
    gradients: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    """Each recorded `gradient.<name>`'s head parameter, by its path in the head's tree."""
    blocks: bool = False
    """Whether the release reads its rows in blocks, whose numbering its engine recorded (`blocks`)."""
    attention_impl: str = "reference"
    """The attention the release is loaded with, as its own tests load it."""
    every_slot: bool = False
    """Whether the recording states its scores at every option slot `[R, Q, O]`, 0 past the real options,
    rather than at the real ones alone."""


def assert_answers_as_its_engine(parity: ReleaseParity, scratch: Path) -> None:
    """Load `parity`'s release through Dew's public loader (`Pretrained.load`, `Decide.of`), lay each
    seed's requests out, a request's recorded pictures before its state by the release's own processor,
    assert the rows the engine laid exactly (tokens, option spans, blocks), and judge its scores and the
    gradients the recording holds by check 1."""
    from PIL import Image

    from dew.decision import Decide, DecisionInputs, Question
    from dew.decision.images import Images
    from dew.interop import Pretrained

    # A release of an engine outside Dew states no task of Dew's: its bundle's (`Decide.of`).
    task = Decide.of(Pretrained.load(parity.release(scratch), dtype="float32",
                                     attention_impl=parity.attention_impl))
    seeds = declared_seeds(parity.recording)[parity.site]
    draw, reference, truth = indexed_recording(recording(parity.recording), parity.site, seeds)
    requests = json.loads(parity.requests.read_text())

    @jax.jit
    def scored(inputs, media=None):
        return task.model.logits(task.variables, inputs, media=media)

    def batched(rows, shown, tokens, mask) -> ModelInputs:
        # The backbone's inputs for a seed's rows of pictures as its engine batched them: its padded
        # tokens, each row's per-token fields over its row and 0 past it, and the pixel fields joined in
        # the processor's batch order.
        assert task.processor is not None, "a release whose engine read pictures loads with its processor"
        aligned = {name: np.concatenate([
            np.pad(images.spread(row)[name], ((0, 0), (0, tokens.shape[1] - len(row.tokens))))
            for row, images in zip(rows, shown, strict=True)]) for name in shown[0].aligned}
        fields = {name: np.concatenate([images.fields[name] for images in shown]) for name in shown[0].fields}
        return task.processor.from_hf({"input_ids": tokens, "attention_mask": mask, **aligned, **fields})

    @jax.jit
    def head_alone(inputs, states):
        return task.model.head_logits(task.variables, states, inputs)[0]

    @jax.jit
    def forward(inputs, probe):
        states = task.model.states(task.variables, inputs)
        held = dict(task.variables)

        def observed(params, hidden):
            # The head over `hidden` as the model calls it (`DecisionModel.head_logits`), its parameters
            # these; the probe's sum over the real options.
            variables = {**held, "params": {**held["params"], "head": params}}
            scores, _ = task.model.head_logits(variables, hidden, inputs)
            return jnp.sum(jnp.where(inputs.options, scores, 0) * probe)

        gradient, hidden = jax.grad(observed, (0, 1))(part(task.variables, "head").get("params", {}), states)

        def at(tree, path):
            for key in path:
                tree = tree[key]
            return tree

        return {"scores": task.model.logits(task.variables, inputs), "gradient.hidden": hidden,
                **{f"gradient.{name}": at(gradient, path) for name, path in parity.gradients.items()}}

    @as_given
    def dew(index, held):
        # A recording of one question a row states it `[R, O]`; every question its own `[R, Q, O]`.
        options = held["options"] if held["options"].ndim == 3 else held["options"][:, None]
        spans = held["spans"] if held["spans"].ndim == 4 else held["spans"][:, None]
        rows, shown = [], []
        for at, request in enumerate(requests[int(index)]):
            questions = {name: Question.from_wire(wire) for name, wire in request["questions"].items()}
            if "images" in held:
                assert task.processor is not None
                pictures = held["images"][at, :held["image_counts"][at]]
                shown.append(Images.of(task.processor, [Image.fromarray(picture) for picture in pictures],
                                       task.processor.image_text()))
            rows.extend(task.layout.rows(task.tokenizer, task.specials, request["state"], questions,
                                         media=shown[-1].tokens if shown else ()))
        laid = DecisionInputs.collate(rows, task.specials.pad, questions=options.shape[1],
                                      options=options.shape[2])
        mask = np.arange(held["tokens"].shape[1])[None] < held["lengths"][:, None]
        # A release read in blocks numbers them as its engine did; one read whole leaves its engine's
        # numbering unread.
        blocked = parity.blocks
        for row, encoded in enumerate(rows):
            np.testing.assert_array_equal(encoded.tokens, held["tokens"][row, :held["lengths"][row]])
            assert len(encoded.questions) == int(np.sum(np.any(options[row], axis=-1)))
            for at, laid_question in enumerate(encoded.questions):
                np.testing.assert_array_equal(laid_question.options, spans[row, at][options[row, at]])
            if blocked:
                assert encoded.blocks is not None
                np.testing.assert_array_equal(encoded.blocks, held["blocks"][row, :held["lengths"][row]])
        inputs = replace(laid, tokens=jnp.asarray(held["tokens"]), segments=jnp.asarray(mask, jnp.int32),
                         positions=jnp.broadcast_to(jnp.arange(mask.shape[1]), mask.shape),
                         blocks=jnp.asarray(held["blocks"]) if blocked else None)
        if "states" in held:
            # The head alone over the engine's own states.
            result = {"scores": head_alone(inputs, jnp.asarray(held["states"]))}
        elif "probe" in held:
            probe = np.zeros(options.shape, np.float32)
            probe[options] = held["probe"]
            result = forward(inputs, jnp.asarray(probe))
        elif shown:
            result = {"scores": scored(inputs, batched(rows, shown, held["tokens"], mask))}
        else:
            result = {"scores": scored(inputs)}
        scores = np.asarray(result["scores"])
        assert np.isneginf(scores[~options]).all()
        return {**result, "scores": np.where(options, scores, 0) if parity.every_slot else scores[options]}

    assert_as_exact(dew, reference, draw, f"{parity.name} as its engine answers", seeds=seeds,
                    truth_fn=truth)
