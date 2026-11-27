"""A template's trace-time derivations stored across processes (`dew.algebra.stored`): a derivation read back
is the one derived, every term, appended parameter and literal of it (its values the evaluator's of those
alone); a corrupt entry is derived afresh past; the
extractor's oracle neither reads nor writes; each outcome is counted (`report.stored`)."""

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

from dew._cache import ALGEBRA
from dew.algebra import SATURATION, derive, stored, terms
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
    assert [(primitive, repr(params)) for primitive, params in read.region.params] == [
        (primitive, repr(params)) for primitive, params in fresh.region.params]
    for held, again in zip(fresh.region.literals, read.region.literals, strict=True):
        np.testing.assert_array_equal(np.asarray(again), np.asarray(held))


@pytest.mark.parametrize("corrupt", ["unknown constructor", "missing steps", "truncated", "a list",
                                     "a merge's side in a step", "a short part", "unreadable",
                                     "an argument of another sort"])
def test_a_corrupt_entry_is_derived_afresh_and_counted(storing, corrupt):
    """An entry that fails to read, or whose forms are not its region's, leaves its region as traced and is
    derived afresh past, then written whole again."""
    fresh = _derived("walk")
    (entry,) = Path(stored.directory()).glob("*.json")
    record = json.loads(entry.read_text())
    stepped = json.dumps(record["steps"][0])
    corrupted = {"unknown constructor": json.dumps({**record, "steps": [["no such", []]]}),
                 "missing steps": json.dumps({**record, "steps": []}),
                 "truncated": entry.read_text()[:100], "a list": "[]",
                 # A step reading the other walk's state, which only a merge carries.
                 "a merge's side in a step": json.dumps({**record, "steps": [
                     json.loads(stepped.replace('["state", [["i", 0], ["i", 0]]]',
                                                '["state", [["i", 0], ["i", 1]]]')),
                     *record["steps"][1:]]}),
                 "a short part": json.dumps({**record, "accumulated": [[] for _ in record["accumulated"]]}),
                 # A convert of an integer where its constructor takes a term.
                 "an argument of another sort": json.dumps({**record, "steps": [
                     ["convert", [["i", 0], ["i", 0]]], *record["steps"][1:]]})}
    if corrupt == "unreadable":
        entry.unlink()
        entry.mkdir()  # an entry's path no file
    else:
        if corrupt == "a merge's side in a step":
            assert '["state", [["i", 0], ["i", 0]]]' in stepped
        entry.write_text(corrupted[corrupt])
    again = _derived("walk")
    # Derived afresh past it, and written again (a directory in its place: not written, `unstorable`).
    assert storing.stored == ({"miss": 1, "invalid": 1, "unstorable": 1} if corrupt == "unreadable"
                              else {"miss": 2, "invalid": 1})
    assert _terms(again) == _terms(fresh)
    assert len(again.region.params) == len(fresh.region.params)
    if corrupt != "unreadable":  # a directory in the entry's place is not written over
        assert _terms(_derived("walk")) == _terms(fresh)
        assert storing.stored == {"miss": 2, "invalid": 1, "hit": 1}


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

    stored.stored("unstated", made.region, len(made.region.params), len(made.region.literals), unstated)
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
