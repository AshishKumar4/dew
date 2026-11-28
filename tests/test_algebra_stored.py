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
    return entry, json.loads(entry.read_text().partition("\n")[2])


def _written(entry, record):
    """`record` written as `entry`'s, digested as its own: what the decoder alone checks."""
    body = json.dumps(record)
    entry.write_text(f"{stored._summed(entry.stem, body)}\n{body}")


RETAGGED = {
    # A walk's step's peak update retagged as a rescale, where the walk's declaration states none (attention's
    # plain form): over a subtraction's parameters (it would bind a subtraction for the maximum), and over the
    # walk's own exponential's (refused for the declaration alone). And the accurate form's own rescale, which
    # its declaration states, over a subtraction's parameters, no exponential's.
    "a peak retagged a rescale over a subtraction": ("walk", "sub"),
    "a peak retagged a rescale over its exponential": ("walk", "exp2"),
    "a rescale reading a subtraction's parameters": ("walk natural", "sub"),
}
"""Each rescale corruption: the declaration whose entry it corrupts, and the operation whose parameters the
rescale reads."""


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
    if corrupt == "a product of another operation's parameters":
        # A product reading an addition's or a product's parameters by index, in range, of another kind.
        at = next(at for at, entry in enumerate(table) if entry[0] == "dot")
        other = next(entry[1][-1] for entry in table
                     if entry[0] == "binary" and entry[1][1] != ["s", terms.RESCALE])
        name, arguments = table[at]
        return {**record, "terms": [*table[:at], [name, [*arguments[:-1], other]], *table[at + 1:]]}
    if corrupt in RETAGGED:
        declared, over = RETAGGED[corrupt]
        if declared == "walk natural":
            # The walk's own rescale.
            at = next(at for at, entry in enumerate(table)
                      if entry[0] == "binary" and entry[1][1] == ["s", terms.RESCALE])
        else:
            # The running peak a step updates (a maximum of its state), not the declaration's own maximum.
            at = next(at for at in record["steps"]
                      if table[at][0] == "binary" and table[at][1][1] == ["s", "max"])
        index = next(entry[1][-1] for entry in table
                     if entry[0] in ("binary", "unary") and entry[1][1] == ["s", over])
        left, _, right, _ = table[at][1]
        return {**record, "terms": [*table[:at], ["binary", [left, ["s", terms.RESCALE], right, index]],
                                    *table[at + 1:]]}
    if corrupt == "a report's admitted sets a string":
        return {**record, "report": [record["report"][0], "sets", *record["report"][2:]]}
    if corrupt == "a finish dropped":
        return {**record, "finishes": [None for _ in record["finishes"]]}
    raise AssertionError(corrupt)


@pytest.mark.parametrize("corrupt", ["unknown constructor", "missing steps", "truncated", "a list",
                                     "a merge's side in a step", "a short part", "unreadable",
                                     "an argument of another sort", "a term of its own", "a float past range",
                                     "a dtype of objects", "an untyped empty vector", "a finish dropped",
                                     "a literal no array", "a product of another operation's parameters",
                                     "a report's admitted sets a string", *RETAGGED,
                                     "nested past depth", "a flipped byte", "another entry's"])
def test_a_corrupt_entry_is_derived_afresh_and_counted(storing, corrupt):
    """An entry that fails to read, is not its own, or whose forms are not its region's, leaves its region as
    traced and is derived afresh past, then written whole again."""
    name = RETAGGED[corrupt][0] if corrupt in RETAGGED else "walk"
    fresh = _derived(name)
    entry, record = _entry()
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
    again = _derived(name)
    # Derived afresh past it, and written again (a directory in its place: not written, `unstorable`).
    assert storing.stored == ({"miss": 1, "invalid": 1, "unstorable": 1} if corrupt == "unreadable"
                              else {"miss": 2, "invalid": 1})
    assert _terms(again) == _terms(fresh)
    assert len(again.region.params) == len(fresh.region.params)
    assert len(again.region.shapes) == len(fresh.region.shapes)
    if corrupt != "unreadable":  # a directory in the entry's place is not written over
        assert _terms(_derived(name)) == _terms(fresh)
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
    decoded = stored._decoded(json.loads(json.dumps(writer.table)), leaves=1, literals=0,
                              primitives=[jax.lax.add_p], states=0)
    assert terms.key(decoded(at, sides=())) == terms.key(form)


def test_a_primitive_its_name_does_not_identify_keys_nothing():
    """A primitive of a user's own named as one of JAX's, which the region reads apart from JAX's by identity
    (`terms.OPERATIONS`), is no primitive a name states: a program holding it, at any depth, is not keyed."""
    own = jcore.Primitive("exp")
    own.def_abstract_eval(lambda operand: operand)
    operand, taken = np.ones(4, np.float32), np.True_

    def branched(function):
        return jax.make_jaxpr(lambda p, x: jax.lax.cond(p, function, jax.lax.neg, x))(taken, operand)

    assert _keyed(jax.make_jaxpr(own.bind)(operand), ()) is None
    assert _keyed(branched(own.bind), ()) is None
    assert _keyed(jax.make_jaxpr(jax.lax.exp)(operand), ()) is not None
    assert _keyed(branched(jax.lax.exp), ()) is not None


def test_a_program_capturing_an_outer_traces_value_has_no_key():
    """A declaration traced inside another trace, closing over its value, keys nothing (no device transfer,
    no error): its derivation is not stored."""
    def outer(scale):
        closed = jax.make_jaxpr(lambda x: x * scale)(np.ones(4, np.float32))
        assert _keyed(closed, ()) is None
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

    named = [_keyed(program(np.full(4, value, np.float32)), ()) for value in (1.0, 2.0)]
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
        stored.encoded_value(object())


WALK = """
import json, os, sys
sys.path[:0] = ["src", "tests", "."]
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


SIBLINGS = """
import json, os, sys
sys.path[:0] = ["src", "tests", "."]
import lane_environment
lane_environment.configure_lane(os.environ)
import numpy as np
from dew.nn.kernels.templates import attention_algebra, report
values = {}
with report.recording() as recorded:
    forms = attention_algebra.derived_forms(True, False, 0.125)
    for rows, keys in json.loads(sys.argv[1]):
        rng = np.random.default_rng(rows)
        tiles = {"logits": rng.standard_normal((rows, keys)).astype(np.float32),
                 "values": rng.standard_normal((keys, 128)).astype(np.float32)}
        shapes = {name: (tile.shape, tile.dtype) for name, tile in tiles.items()}
        state = forms.step(forms.init(shapes), tiles)
        values[f"{rows}x{keys}"] = [np.asarray(part).tobytes().hex()
                                    for part in (*state, *forms.finish(state, tiles, shapes=shapes))]
print(json.dumps({"stored": dict(recorded.stored), "values": values}))
"""


def test_a_class_stored_by_one_process_serves_another_whichever_size_each_takes_first(tmp_path):
    """Two tiles of one class of sizes (12 by 24 and 13 by 20 logits: canonical 16 by 32), walked in one
    process in one order and in another, of another hash seed, in the other, each as a run computes (the CPU's
    wide arithmetic off, whose wide forms key no class of this declaration): the first derives the class's
    canonical sizes once and stores them, the second reads that entry and derives none, and each tile's walk
    is the same bit for bit in both."""
    from dew.nn.precision import WIDE_CPU_ARITHMETIC

    def walked(seed, order):
        # Outside the test session, the extractor's oracle is off, as in a run.
        env = {**os.environ, "XDG_CACHE_HOME": str(tmp_path), "PYTHONHASHSEED": seed,
               WIDE_CPU_ARITHMETIC: "0"}
        done = subprocess.run([sys.executable, "-c", SIBLINGS, json.dumps(order)], capture_output=True,
                              text=True, env=env, check=True)
        return json.loads(done.stdout.strip().splitlines()[-1])

    first = walked("1", [[12, 24], [13, 20]])
    second = walked("2", [[13, 20], [12, 24]])
    assert first["stored"] == {"miss": 1}, first["stored"]
    assert second["stored"] == {"hit": 1}, second["stored"]
    assert second["values"] == first["values"]


def test_a_replacements_stored_outcomes_are_the_kept_programs_whether_or_not_it_is_kept():
    """The store's reading or writing a replacement's trace made (`report._proposing`) is recorded in the
    kept program's registry and `recording` at once: a replacement discarded keeps them, one kept
    (`report._commit`) records them once."""
    with kernel_report.refusals(), kernel_report.recording() as recorded:
        with kernel_report._proposing([]):
            kernel_report.record_stored("hit")
        assert recorded.stored == {"hit": 1} and kernel_report.stored() == {"hit": 1}
        with kernel_report._proposing([]) as proposal:
            kernel_report.record_stored("miss")
        kernel_report._commit(proposal)
        assert recorded.stored == {"hit": 1, "miss": 1} and kernel_report.stored() == {"hit": 1, "miss": 1}


def test_a_proposals_stored_outcomes_reach_the_scopes_opened_within_it():
    """Within a proposal, a `recording` opened there counts the store's outcomes too, a `refusals` scope
    opened there counts its own apart from the registry's, and a nested proposal discarded counts its own in
    the kept program's."""
    with kernel_report.refusals(), kernel_report.recording() as outer:
        with kernel_report._proposing([]):
            with kernel_report.recording() as inner:
                kernel_report.record_stored("hit")
            with kernel_report.refusals():
                kernel_report.record_stored("miss")
                assert kernel_report.stored() == {"miss": 1}
            with kernel_report._proposing([]):
                kernel_report.record_stored("invalid")
        assert inner.stored == {"hit": 1}
        assert outer.stored == {"hit": 1, "miss": 1, "invalid": 1}
        assert kernel_report.stored() == {"hit": 1, "invalid": 1}


def test_a_fused_multiply_adds_terms_read_as_the_operations_they_bind():
    """A fused multiply-add's terms (`rules.CONTRACTED`), named apart from the multiply and add whose
    parameters they bind, read back over those; over another operation's parameters, refused."""
    operand = Term.leaf(i64(0))
    form = operand.binary(String("fmul"), operand, i64(0)).binary(String("fadd"), operand, i64(1))
    writer = stored._Writer()
    at = writer.term(form)
    table = json.loads(json.dumps(writer.table))
    decoded = stored._decoded(table, leaves=1, literals=0, primitives=[jax.lax.mul_p, jax.lax.add_p],
                              states=0)
    assert terms.key(decoded(at, sides=())) == terms.key(form)
    with pytest.raises(ValueError):
        stored._decoded(table, leaves=1, literals=0, primitives=[jax.lax.add_p, jax.lax.add_p], states=0)


@pytest.fixture
def shipped(tmp_path, monkeypatch):
    """A read-only store of the test's own in the package's place (`DEW_DERIVATIONS`), not made yet."""
    monkeypatch.setenv("DEW_DERIVATIONS", str(tmp_path / "shipped"))
    return tmp_path / "shipped"


def _ship(shipped):
    """The store's entries moved into the read-only one, the store left empty: a fresh install's."""
    shipped.mkdir(exist_ok=True)
    for entry in Path(stored.directory()).glob("*.json"):
        entry.rename(shipped / entry.name)


def test_a_shipped_derivation_reads_into_an_empty_store_and_writes_nothing(storing, shipped):
    fresh = _derived("walk")
    _ship(shipped)
    read = _derived("walk")
    assert storing.stored == {"miss": 1, "hit": 1}
    assert _terms(read) == _terms(fresh)
    assert not list(Path(stored.directory()).glob("*.json"))


def test_the_store_is_read_before_the_shipped_entries(storing, shipped):
    """Where both hold an entry, Dew's cache's is read: the shipped one, here no entry at all, is not."""
    fresh = _derived("walk")
    entry, _ = _entry()
    shipped.mkdir()
    (shipped / entry.name).write_text("no entry")
    read = _derived("walk")
    assert storing.stored == {"miss": 1, "hit": 1}
    assert _terms(read) == _terms(fresh)


def test_shipped_entries_another_build_derived_are_derived_afresh_past(storing, shipped):
    """Shipped entries keyed otherwise (another derivation's, or another build's code, egglog or JAX) are
    none of a derivation's: it is derived afresh and written to Dew's cache. One shipped under its name but
    not its own (another's text) is counted invalid and derived afresh past."""
    _derived("weights")
    _ship(shipped)
    (other,) = shipped.glob("*.json")
    fresh = _derived("walk")
    assert storing.stored == {"miss": 2}
    entry, _ = _entry()
    entry.unlink()
    (shipped / entry.name).write_text(other.read_text())
    again = _derived("walk")
    assert storing.stored == {"miss": 3, "invalid": 1}
    assert _terms(again) == _terms(fresh)
    assert [path.name for path in Path(stored.directory()).glob("*.json")] == [entry.name]


def _manual(sizes, names=("data",)):
    """An abstract mesh of `sizes` over manual axes `names`, a sharded step's kernels' as they trace."""
    return jax.sharding.use_abstract_mesh(jax.sharding.AbstractMesh(
        sizes, names, axis_types=(jax.sharding.AxisType.Manual,) * len(names)))


def test_a_derivation_stored_on_one_device_is_read_on_eight(storing):
    """A walk derived on a manual mesh of one device is read on one of eight: a key states a mesh by its
    axes' names and types, which a derivation reads, and none of their sizes, which it does not."""
    with _manual((1,)):
        fresh = _derived("walk")
    with _manual((8,)):
        read = _derived("walk")
    assert storing.stored == {"miss": 1, "hit": 1}
    assert _terms(read) == _terms(fresh)


def test_a_derivation_on_a_manual_mesh_is_none_of_one_on_no_mesh(storing):
    """Whether a mesh's axes are manual is read by a derivation (a tangent sum, an fp32-exact product and a
    cast's round trip are each treated by it, `terms.unplaced_equation`): one stored on a manual mesh is
    derived afresh on none."""
    with _manual((8,)):
        _derived("walk")
    _derived("walk")
    assert storing.stored == {"miss": 2}


def test_a_mesh_axis_named_as_an_address_keys_its_derivations(storing):
    """A mesh axis whose name reads as an object's address is a name like any other: the derivations on it
    are stored and read back."""
    with _manual((2,), names=("data at 0x1",)):
        _derived("walk")
        _derived("walk")
    assert storing.stored == {"miss": 1, "hit": 1}


def _keyed(closed, parts=()):
    """The store's key of `closed`'s derivation over its region (`terms.region`), with no rule sets."""
    return stored.key(closed, terms.region(closed), [], parts)


def _mesh(sizes):
    return jax.sharding.AbstractMesh(sizes, ("a", "b"), axis_types=(jax.sharding.AxisType.Explicit,) * 2)


def _broadcasts(sizes, moved=None, to=(2, 1)):
    """Three broadcasts of one value, summed, traced on an Explicit mesh of `sizes`, each placing its result
    on it; the `moved`th's parameters placed on a mesh of the same axes of sizes `to` instead."""
    mesh = _mesh(sizes)
    where = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    def program(x):
        return sum(jax.lax.broadcast_in_dim(x, (2, 4), (1,), out_sharding=where) for _ in range(3))

    with jax.sharding.use_abstract_mesh(mesh):
        closed = jax.make_jaxpr(program)(jax.ShapeDtypeStruct((4,), jnp.float32))
    if moved is None:
        return closed
    broadcasts = [at for at, equation in enumerate(closed.jaxpr.eqns)
                  if equation.primitive is jax.lax.broadcast_in_dim_p]
    elsewhere = jax.sharding.NamedSharding(_mesh(to), jax.sharding.PartitionSpec())
    equations = [equation.replace(params={**equation.params, "sharding": elsewhere})
                 if at == broadcasts[moved] else equation for at, equation in enumerate(closed.jaxpr.eqns)]
    return closed.replace(jaxpr=closed.jaxpr.replace(eqns=equations))


def test_a_key_states_a_mesh_by_name_and_how_its_parameters_intern():
    """A stored derivation's key states the meshes its program reads by their axes' names and types, not
    their sizes, beside how its region interns its parameters (`Region.interned`): a program traced on a
    mesh of one device and of four keys alike, its parameters' placements among them; two whose
    parameters are one but for their meshes' sizes and intern apart (a broadcast placed on another mesh
    of those axes, the second or the third) key apart."""
    assert _keyed(_broadcasts((1, 1), None), ()) == _keyed(_broadcasts((2, 2), None), ())
    second, third = (_keyed(_broadcasts((1, 2), at), ()) for at in (1, 2))
    assert None not in (second, third) and second != third


def _composed(order):
    """A scalar broadcast to (1,) and on to (2, 4) along each axis, broadcast to (3,), and broadcast to (2, 4)
    directly: the three (2, 4) broadcasts placed on the meshes of `order`'s sizes in turn (one axis's apart),
    which a region interns after the composed broadcasts' dimensions are replaced by the direct one's."""
    def program(x):
        one = jax.lax.broadcast_in_dim(x, (1,), ())
        return (jax.lax.broadcast_in_dim(one, (2, 4), (0,)), jax.lax.broadcast_in_dim(one, (2, 4), (1,)),
                jax.lax.broadcast_in_dim(x, (3,), ()), jax.lax.broadcast_in_dim(x, (2, 4), ()))

    closed = jax.make_jaxpr(program)(jax.ShapeDtypeStruct((), jnp.float32))
    placed = iter(jax.sharding.NamedSharding(_mesh(sizes), jax.sharding.PartitionSpec()) for sizes in order)
    equations = [equation.replace(params={**equation.params, "sharding": next(placed)})
                 if equation.params["shape"] == (2, 4) else equation for equation in closed.jaxpr.eqns]
    return closed.replace(jaxpr=closed.jaxpr.replace(eqns=equations))


@pytest.mark.parametrize("other", [((1, 1), (1, 1), (2, 1)), ((1, 1), (2, 1), (2, 1))],
                         ids=["slots apart", "outputs apart"])
def test_a_key_states_the_parameters_its_region_interns(other):
    """A key states how the region interns its parameters, after it normalizes them, and which output reads
    which: two programs one but for which (2, 4) broadcast is placed on the mesh of another size key apart,
    whether their regions' parameter slots differ or only which output reads a slot (head reviews' cases)."""
    first, second = _keyed(_composed(((1, 1), (2, 1), (1, 1)))), _keyed(_composed(other))
    assert None not in (first, second) and first != second


def test_a_part_of_objects_keys_nothing():
    """A part whose value is held by identity, an array of objects among them (its bytes their addresses),
    keys no derivation."""
    assert _keyed(_broadcasts((1, 1)), (np.array([object()], dtype=object),)) is None


def test_a_read_only_store_one_directory_with_dews_cache_by_another_path_is_never_written(storing, tmp_path,
                                                                                        monkeypatch):
    """A read-only store that is Dew's cache's directory by another path no resolution joins (two mounts of
    one) is never written: here a link whose resolution the test withholds, so the two paths differ and only
    the directories' identity (`os.path.samefile`) tells them one."""
    cache = Path(stored.directory())
    cache.mkdir(parents=True)
    (tmp_path / "mount").symlink_to(cache)
    monkeypatch.setenv("DEW_DERIVATIONS", str(tmp_path / "mount"))
    withheld = {os.path.abspath(path) for path in (cache, tmp_path / "mount")}
    resolved = os.path.realpath

    def realpath(path, *args, **kwargs):
        held = os.path.abspath(path)
        return held if held in withheld else resolved(path, *args, **kwargs)

    monkeypatch.setattr(os.path, "realpath", realpath)
    _derived("walk")
    assert storing.stored == {"unstorable": 1}
    assert not list(cache.glob("*.json"))


@pytest.mark.parametrize("alias", ["named", "linked", "not yet made"])
def test_a_read_only_store_that_is_dews_cache_is_never_written(storing, tmp_path, monkeypatch, alias):
    """A read-only store (`DEW_DERIVATIONS`) that is Dew's cache, named, through a link, or named before
    either is made, is never written: a derivation there is derived and counted unstorable, the store
    left empty."""
    cache = Path(stored.directory())
    if alias != "not yet made":
        cache.mkdir(parents=True, exist_ok=True)
    if alias == "linked":
        (tmp_path / "alias").symlink_to(cache)
    monkeypatch.setenv("DEW_DERIVATIONS", str(tmp_path / "alias" if alias == "linked" else cache))
    _derived("walk")
    assert storing.stored == {"unstorable": 1}
    assert not list(cache.glob("*.json"))


def test_an_entry_is_a_stores_own_file_not_its_name(storing, tmp_path):
    """A file named as an entry is one only where its text is its own (`stored.entry`), as the catalogue's
    output is replaced only where it holds entries alone."""
    _derived("walk")
    entry, _ = _entry()
    assert stored.entry(entry)
    named = tmp_path / entry.name
    named.write_text('{"format": 5}')
    assert not stored.entry(named) and not stored.entry(tmp_path / "missing.json")
    # Its digest read before its body: a body nested past Python's depth, not its own, is never parsed.
    named.write_text("0" * 40 + "\n" + "[" * 100_000 + "]" * 100_000)
    assert not stored.entry(named)
