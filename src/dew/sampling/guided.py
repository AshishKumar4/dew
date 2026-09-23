"""Guided decoding: draws held to a regular expression or a JSON schema.

`outlines-core` compiles the pattern against the tokenizer's vocabulary into a
token-level automaton, once, on the host: a state per prefix of the
language, and for each state the tokens that keep the text inside it and
the state each one leads to. `Grammar` holds that automaton as two device
tables, and `Sample(grammar)` (`dew.sampling.strategies`) carries each
row's state through its loop: before every draw the row's disallowed tokens
score -inf, and the drawn token moves the row's state on. A state that
completes the language allows EOS, so a row ends where the pattern may.

Most tokens behave alike in every state (any letter inside a JSON string,
say), so the tables are compact: tokens with the same column of the
transition table share a class, and the device table is `[states,
classes]` with a `[vocab]` class per token. Class 0 is allowed nowhere; the
ids the tokenizer does not spell (special tokens, a checkpoint's padded
head) take it.

JSON schemas compile through `outlines_core.json_schema.build_regex_from_schema`,
which is the schema-to-regex translation outlines and vLLM use.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax.experimental import checkify

from dew.records import JSON
from dew.sampling.decoding import (
    FILTER,
    Referencing,
    StepState,
    Tokenizing,
    Vocabulary,
    matching_mode,
    vocabulary_pieces,
)


@struct.dataclass
class Grammar:
    """A token automaton on the device.

    `transitions[state, class]` is the state a token of that class leads to,
    -1 where the token is not allowed; `classes[token]` is the token's
    class. State 0 is the state before the first draw, so a zeroed carry
    starts a row.
    """

    transitions: jax.Array
    classes: jax.Array

    def start(self, rows: int) -> jax.Array:
        return jnp.zeros((rows,), jnp.int32)

    def masked(self, state: jax.Array, logits: jax.Array) -> jax.Array:
        """`logits` `[rows, vocab]` with the tokens each row's state forbids at -inf."""
        allowed = jnp.take(self.transitions[state], self.classes, axis=1) >= 0
        return jnp.where(allowed, logits, FILTER)

    def guiding(self, transform: Callable[[StepState, jax.Array], jax.Array],
                state: jax.Array) -> Callable[[StepState, jax.Array], jax.Array]:
        """`transform` behind the mask of each row's state."""
        return lambda step, logits: transform(step, self.masked(state, logits))

    def advanced(self, state: jax.Array, token: jax.Array, drawn: jax.Array) -> jax.Array:
        """Each row's state after `token`; a row that did not draw keeps its state.

        A drawn token the state forbids can only come from a transform that
        forces a token after the mask (`ForcedEOS`, say), and fails the
        device check rather than leaving the text outside the language.
        """
        following = self.transitions[state, self.classes[token]]
        checkify.check(jnp.all(~drawn | (following >= 0)),
                       "a transform forced a token the grammar forbids")
        return jnp.where(drawn, following, state)


@runtime_checkable
class Special(Protocol):
    """A Transformers tokenizer's special-token ids, which spell no text."""

    @property
    def all_special_ids(self) -> list[int]: ...


def _vocabulary(tokenizer: Vocabulary | Referencing | Tokenizing) -> Vocabulary:
    """The tokenizer beneath a processor, as `stop_strings` finds it."""
    source = tokenizer
    for _ in range(3):
        if isinstance(source, Vocabulary):
            return source
        if isinstance(source, Referencing):
            source = source.reference
        elif isinstance(source, Tokenizing):
            source = source.tokenizer
    if not isinstance(source, Vocabulary):
        raise TypeError("guided decoding needs a tokenizer that can list its vocabulary")
    return source


def regex(tokenizer: Vocabulary | Referencing | Tokenizing, pattern: str, eos_id: int | Sequence[int],
          vocab_size: int | None = None) -> Grammar:
    """The automaton of `pattern` over the tokenizer's vocabulary.

    `eos_id` is the id, or ids, that end a row, as `Sampling.eos_id` names
    them; a state that completes the pattern allows them. `vocab_size`
    sizes the class table for the model's head when it pads past the
    tokenizer.
    """
    from outlines_core import Index, Vocabulary as Pieces

    stops = (eos_id,) if isinstance(eos_id, int) else tuple(eos_id)
    if not stops:
        raise ValueError("guided decoding needs an EOS id to end a row on")
    source = _vocabulary(tokenizer)
    pieces, ids = vocabulary_pieces(source, matching_mode(source))
    special = set(source.all_special_ids) if isinstance(source, Special) else set()
    spelled: dict[str | bytes, list[int]] = {}
    for piece, index in zip(pieces, ids, strict=True):
        if index not in special and index not in stops and piece:
            spelled.setdefault(piece, []).append(index)
    index = Index(pattern, Pieces(stops[0], spelled))
    width = max(max(ids) + 1, vocab_size or 0)
    return _tables(index.get_transitions(), index.get_initial_state(), index.get_final_states(),
                   stops, width)


def json_schema(tokenizer: Vocabulary | Referencing | Tokenizing, schema: str | Mapping[str, JSON],
                eos_id: int | Sequence[int], vocab_size: int | None = None,
                whitespace: str | None = None) -> Grammar:
    """The automaton of the JSON documents `schema` accepts; see `regex`.

    `whitespace` is the regex the separators may match, None taking
    outlines' default of at most one space.
    """
    from outlines_core.json_schema import build_regex_from_schema

    text = schema if isinstance(schema, str) else json.dumps(schema)
    return regex(tokenizer, build_regex_from_schema(text, whitespace), eos_id, vocab_size)


def _tables(transitions: Mapping[int, Mapping[int, int]], initial: int, finals: set[int],
            stops: tuple[int, ...], width: int) -> Grammar:
    """Number the states densely and group tokens with equal columns into classes."""
    names = [initial] + sorted((set(transitions) | set(finals)
                                | {state for row in transitions.values() for state in row.values()})
                               - {initial})
    number = {name: order for order, name in enumerate(names)}
    columns: dict[int, list[tuple[int, int]]] = {}
    for state, row in transitions.items():
        for token, following in row.items():
            if token < width:
                columns.setdefault(token, []).append((number[state], number[following]))
    for final in finals:
        for stop in stops:
            if stop < width:
                entry = (number[final], number[final])
                column = columns.setdefault(stop, [])
                if entry not in column:
                    column.append(entry)
    classes = np.zeros(width, np.int32)
    signatures: dict[tuple[tuple[int, int], ...], int] = {}
    for token, column in columns.items():
        classes[token] = signatures.setdefault(tuple(sorted(column)), len(signatures) + 1)
    table = np.full((len(names), len(signatures) + 1), -1, np.int32)
    for signature, kind in signatures.items():
        for state, following in signature:
            table[state, kind] = following
    return Grammar(jnp.asarray(table), jnp.asarray(classes))
