"""A template's trace-time derivations stored across processes (`dew.algebra.stored`): a derivation read back
is the one derived, every term, appended parameter, literal and shape of it (its values the evaluator's of
those alone); a corrupt entry, or one not its own, is derived afresh past; a shared term is written once; a
primitive its name does not identify keys nothing; the extractor's oracle neither reads nor writes; each
outcome is counted (`report.stored`)."""

import dataclasses
import functools
import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from egglog import String, i64
from jax.extend import core as jcore

from dew._cache import ALGEBRA
from dew.algebra import SATURATION, derive, stored, terms
from dew.algebra.terms import Term
from dew.nn.kernels.templates import attention_algebra, attention_backward_algebra, report as kernel_report


@pytest.fixture
def storing(tmp_path, monkeypatch):
    """A store of the test's own, the oracle off (as a run's): the counts its derivations made."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    with ALGEBRA(dataclasses.replace(ALGEBRA.value, checked=False)), kernel_report.recording() as recorded:
        yield recorded


DECLARATIONS = {
    "walk": (attention_algebra.declaration(base2=True, factor=0.125), {"logits": ((64, 64), jnp.float32),
                                                                     "values": ((64, 128), jnp.float32)}),
    "walk natural": (attention_algebra.declaration(base2=False, natural=True),
                     {"logits": ((64, 64), jnp.float32), "values": ((64, 64), jnp.float32)}),
    # The backward's weights (`attention_backward_algebra.derived_weights(False)`'s declaration).
    "weights": (derive.Reduction(functools.partial(attention_backward_algebra.softmax_weights,
                                                   keys_first=False, how=attention_algebra.PLAIN,
                                                   accurate=False),
                                 {"logits": ("rows", "keys"), "peak": ("rows",), "total": ("rows",)},
                                 walked="keys", exponent="e"),
                {"logits": ((64, 64), jnp.float32), "peak": ((64,), jnp.float32),
                 "total": ((64,), jnp.float32)}),
}


def _derived(name):
    declaration, shapes = DECLARATIONS[name]
    names = tuple(declaration.operands)
    walked = {key: axes.index(declaration.walked) for key, axes in declaration.operands.items()
              if declaration.walked in axes}
    return derive._derived(declaration, names, walked, shapes, name == "walk natural", SATURATION)


def _terms(made):
    fields = [*made.reductions, *made.steps, *made.merges, *made.anchors, *made.scales,
              *(form for form in made.finishes if form is not None),
              *(form for parts in (*made.accumulated, *made.added, *made.merged) if parts is not None
                for form in parts if form is not None)]
    return [terms.key(form) for form in fields]


@pytest.mark.parametrize("name", list(DECLARATIONS))
def test_a_stored_derivation_reads_back_as_derived(storing, name):
    fresh = _derived(name)
    assert storing.stored == {"miss": 1}
    read = _derived(name)
    assert storing.stored == {"miss": 1, "hit": 1}
    assert read.report.reused and not fresh.report.reused
    assert _terms(read) == _terms(fresh)
    assert (read.floors, read.mapped, read.walked_leaves) == (fresh.floors, fresh.mapped, fresh.walked_leaves)
    assert len(read.region.params) == len(fresh.region.params)
    assert all(primitive is held and _same(params, values) for (primitive, params), (held, values)
               in zip(read.region.params, fresh.region.params, strict=True))
    assert _same(read.region.literals, fresh.region.literals)
    # Each term's shape, the ones the derivation appended (a literal of its own) too, as the region reads it.
    assert [(terms.key(term), extent) for term, extent in read.region.shapes] == [
        (terms.key(term), extent) for term, extent in fresh.region.shapes]
    assert all(read.region.shape(term) == extent for term, extent in fresh.region.shapes)


def _same(read, held) -> bool:
    """Whether `read` is `held` of the same types at every depth: an array of its dtype, shape and values."""
    if type(read) is not type(held):
        return False
    if isinstance(held, np.ndarray):
        return read.dtype == held.dtype and read.shape == held.shape and np.array_equal(read, held,
                                                                                         equal_nan=True)
    if isinstance(held, (tuple, list)):
        return len(read) == len(held) and all(map(_same, read, held))
    if isinstance(held, dict):
        return read.keys() == held.keys() and all(_same(read[name], held[name]) for name in held)
    return bool(read == held) or (read != read and held != held)


def _entry():
    (entry,) = Path(stored.directory()).glob("*.json")
    summed, _, body = entry.read_text().partition("\n")
    return entry, summed, json.loads(body)


def _written(entry, record):
    """`record` written as `entry`'s, digested as its own: what the decoder alone checks."""
    body = json.dumps(record)
    entry.write_text(f"{stored._summed(entry.stem, body)}\n{body}")


def _corrupted(record, corrupt):
    table = record["terms"]
    if corrupt == "unknown constructor":
        return {**record, "terms": [["no such", []], *table[1:]]}
    if corrupt == "missing steps":
        return {**record, "steps": []}
    if corrupt == "a merge's side in a step":
        # The walk's own state, which a step reads, made the other walk's, which only a merge carries.
        (state,) = (at for at, entry in enumerate(table) if entry == ["state", [["i", 0], ["i", 0]]])
        return {**record, "terms": [*table[:state], ["state", [["i", 0], ["i", 1]]], *table[state + 1:]]}
    if corrupt == "a short part":
        return {**record, "accumulated": [[] for _ in record["accumulated"]]}
    if corrupt == "an argument of another sort":
        # A convert of an integer where its constructor takes a term.
        return {**record, "terms": [*table, ["convert", [["i", 0], ["i", 0]]]],
                "steps": [len(table), *record["steps"][1:]]}
    if corrupt == "a term of its own":
        at = next(at for at, entry in enumerate(table) if any(kind == "t" for kind, _ in entry[1]))
        name, arguments = table[at]
        own = [["t", at] if kind == "t" else [kind, value] for kind, value in arguments]
        return {**record, "terms": [*table[:at], [name, own], *table[at + 1:]]}
    if corrupt == "a float past range":
        return {**record, "report": [*record["report"][:-1], ["x", "0x1p+99999"]]}
    if corrupt == "a dtype of objects":
        return {**record, "literals": [*record["literals"], ["a", "object", [1], "00"]]}
    if corrupt == "an untyped empty vector":
        return {**record, "terms": [*table, ["reduce", [["t", 0], ["s", "add"], ["u", []], ["i", 0]]]],
                "steps": [len(table), *record["steps"][1:]]}
    if corrupt == "a literal no array":
        return {**record, "literals": [*record["literals"], ["T", [["p", 1], ["T", []]]]]}
    if corrupt == "a finish dropped":
        return {**record, "finishes": [None for _ in record["finishes"]]}
    raise AssertionError(corrupt)


@pytest.mark.parametrize("corrupt", ["unknown constructor", "missing steps", "truncated", "a list",
                                     "a merge's side in a step", "a short part", "unreadable",
                                     "an argument of another sort", "a term of its own", "a float past range",
                                     "a dtype of objects", "an untyped empty vector", "a finish dropped",
                                     "a literal no array",
                                     "nested past depth", "a flipped byte", "another entry's"])
def test_a_corrupt_entry_is_derived_afresh_and_counted(storing, corrupt):
    """An entry that fails to read, is not its own, or whose forms are not its region's, leaves its region as
    traced and is derived afresh past, then written whole again."""
    fresh = _derived("walk")
    entry, summed, record = _entry()
    text = entry.read_text()
    if corrupt == "unreadable":
        entry.unlink()
        entry.mkdir()  # an entry's path no file
    elif corrupt == "truncated":
        entry.write_text(text[:100])
    elif corrupt == "a flipped byte":
        at = text.index('"steps"')
        entry.write_text(text[:at] + text[at:].replace("1", "2", 1))
    elif corrupt == "another entry's":
        body = text.partition("\n")[2]
        entry.write_text(f"{stored._summed('another', body)}\n{body}")
    elif corrupt == "a list":
        entry.write_text(f"{stored._summed(entry.stem, '[]')}\n[]")
    elif corrupt == "nested past depth":
        nested = "[" * 100_000 + "]" * 100_000
        body = json.dumps({**record, "literals": None}).replace('"literals": null', f'"literals": {nested}')
        entry.write_text(f"{stored._summed(entry.stem, body)}\n{body}")
    else:
        _written(entry, _corrupted(record, corrupt))
    again = _derived("walk")
    # Derived afresh past it, and written again (a directory in its place: not written, `unstorable`).
    assert storing.stored == ({"miss": 1, "invalid": 1, "unstorable": 1} if corrupt == "unreadable"
                              else {"miss": 2, "invalid": 1})
    assert _terms(again) == _terms(fresh)
    assert len(again.region.params) == len(fresh.region.params)
    assert len(again.region.shapes) == len(fresh.region.shapes)
    if corrupt != "unreadable":  # a directory in the entry's place is not written over
        assert _terms(_derived("walk")) == _terms(fresh)
        assert storing.stored == {"miss": 2, "invalid": 1, "hit": 1}


def test_a_shared_term_is_written_once_and_read_back():
    """A form whose terms share a term, as egglog holds it once, is written once per distinct term, not
    once per path to it: a chain of 64 doublings, 2**64 paths, is 65 entries, and reads back as itself."""
    form = Term.leaf(i64(0))
    for _ in range(64):
        form = form.binary(String("add"), form, i64(0))
    writer = stored._Writer()
    at = writer.term(form)
    assert len(writer.table) == 65
    decoded = stored._decoded(json.loads(json.dumps(writer.table)), leaves=1, literals=0, params=1, states=0)
    assert terms.key(decoded(at, sides=())) == terms.key(form)


def test_a_primitive_its_name_does_not_identify_keys_nothing():
    """A primitive of a user's own named as one of JAX's, which the region reads apart from JAX's by identity
    (`terms.OPERATIONS`), is no primitive a name states: a program holding it, at any depth, is not keyed."""
    own = jcore.Primitive("exp")
    own.def_abstract_eval(lambda operand: operand)
    operand = np.ones(4, np.float32)
    def branched(function):
        return jax.make_jaxpr(lambda p, x: jax.lax.cond(p, function, jax.lax.neg, x))(True, operand)

    assert stored.key(jax.make_jaxpr(own.bind)(operand), (), ()) is None
    assert stored.key(branched(own.bind), (), ()) is None
    assert stored.key(jax.make_jaxpr(jax.lax.exp)(operand), (), ()) is not None
    assert stored.key(branched(jax.lax.exp), (), ()) is not None


def test_a_program_capturing_an_outer_traces_value_has_no_key():
    """A declaration traced inside another trace, closing over its value, keys nothing (no device transfer,
    no error): its derivation is not stored."""
    def outer(scale):
        closed = jax.make_jaxpr(lambda x: x * scale)(np.ones(4, np.float32))
        assert stored.key(closed, (), ()) is None
        return scale

    jax.jit(outer)(np.float32(2))


def test_a_rule_set_the_source_does_not_register_is_not_sourced():
    """A set registered at run time, or declared under a source set's name, is no set the source registers
    (`rules.sourced`), whose name states no rules: a derivation with it is stored nowhere
    (`test_algebra_derive`'s registration witnesses its derivation, afresh with it)."""
    from dew.algebra import rules

    registered = rules.admitted(accurate=False)
    assert rules.sourced(registered)
    runtime = rules.Rules.declared("a test's own", "exact", registered[0].families, "a test's")
    impostor = rules.Rules.declared(registered[0].name, "exact", registered[0].families, "a test's")
    assert not rules.sourced((*registered, runtime)) and not rules.sourced((impostor,))


def test_an_entry_keys_on_its_programs_captured_constants():
    """Two programs of one structure whose captured constants differ, which the region's facts read, key
    apart."""
    def program(constant):
        return jax.make_jaxpr(lambda x: x * constant)(np.ones(4, np.float32))

    named = [stored.key(program(np.full(4, value, np.float32)), (), ()) for value in (1.0, 2.0)]
    assert None not in named and named[0] != named[1]


def test_the_oracle_neither_reads_nor_writes_the_store(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    with kernel_report.recording() as recorded:
        assert ALGEBRA.value.checked
        _derived("weights")
        _derived("weights")
    assert recorded.stored == {}
    assert not os.path.isdir(stored.directory())


def test_a_derivation_the_codec_does_not_state_is_counted_unstorable(storing):
    made = _derived("weights")

    def unstated(**codec):
        raise stored.Unstated("a value of no stored kind")

    region = made.region
    stored.stored("unstated", region, len(region.params), len(region.literals), len(region.shapes), unstated)
    assert storing.stored == {"miss": 1, "unstorable": 1}
    assert not (Path(stored.directory()) / "unstated.json").exists()
    with pytest.raises(stored.Unstated):
        stored._encoded_value(object())


WALK = """
import json, os, sys
sys.path[:0] = ["tests", "."]
import lane_environment
lane_environment.configure_lane(os.environ)
import numpy as np
from dew.nn.kernels.templates import attention_algebra, report
with report.recording() as recorded:
    forms = attention_algebra.derived_forms(True, False, 0.125)
    rng = np.random.default_rng(0)
    tiles = {"logits": rng.standard_normal((64, 64)).astype(np.float32),
             "values": rng.standard_normal((64, 128)).astype(np.float32)}
    shapes = {name: (tile.shape, tile.dtype) for name, tile in tiles.items()}
    state = forms.init(shapes)
    state = forms.step(state, tiles)
    state = forms.merge(state, forms.step(forms.init(shapes), tiles), shapes=shapes)
    out = forms.finish(state, tiles, shapes=shapes)
print(json.dumps({"stored": dict(recorded.stored),
                  "values": [np.asarray(part).tobytes().hex() for part in (*state, *out)]}))
"""


def test_a_fresh_process_walks_a_template_by_the_forms_an_earlier_one_stored(tmp_path):
    """The attention walk's forms derived in one process and read back in another, of another hash seed:
    the second derives none, and its walk's states and outputs are the first's, bit for bit."""
    def walked(seed):
        # Outside the test session, the extractor's oracle is off, as in a run.
        env = {**os.environ, "XDG_CACHE_HOME": str(tmp_path), "PYTHONHASHSEED": seed}
        done = subprocess.run([sys.executable, "-c", WALK], capture_output=True, text=True, env=env,
                              check=True)
        return json.loads(done.stdout.strip().splitlines()[-1])

    first, second = walked("1"), walked("2")
    assert first["stored"].get("miss") and not first["stored"].get("hit")
    assert second["stored"] == {"hit": first["stored"]["miss"]}
    assert second["values"] == first["values"]
