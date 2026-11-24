"""Programs one process compiles at Dew's compile point, kept (`dew.training.ahead.keeping`) and handed to a
later process (`dew.training.ahead.kept`) by their fingerprint: the digest of the lowered program, its trees
and its arguments' types under its compile options (`dew.nn.kernels.options.program_digest`), with the
target they were built for. A handed call traces and lowers as it would and runs the kept executable of the
program it lowers to; any other compiles (training.md §6)."""

import dataclasses
import enum
import functools
import os

import jax
import jax.numpy as jnp
import numpy as np
import process_support
import pytest

KEPT_MODULE = '''
import dataclasses
import os

import jax.numpy as jnp

from dew._facts import traced_fact

OFFSET = 0


@dataclasses.dataclass(frozen=True)
class Scale:
    by: int


def scaled(scale, values):
    return jnp.tanh(values) * scale.by


def shifted(values):
    return values + OFFSET


def chosen(flag, values):
    return values + (1 if flag is None else 2)


def _by(values):
    return values * OFFSET


def helped(values):
    return _by(values)


def nested(values):
    return (lambda held: held - OFFSET)(values)


@dataclasses.dataclass(frozen=True)
class Factor:
    factor = 1

    def __call__(self, values):
        return values * self.factor


def applied(factor, values):
    return factor(values)


@traced_fact
def kept_fact():
    return os.environ.get("KEPT_FACT")
'''
"""The module the kept programs' functions and their static argument's class are defined in
(tests/kept_programs_worker.py), which each test writes beside its file: a user's own, outside any installed
distribution, with a setting its code changes at run time (`OFFSET`) and a fact of the process it declares
its traces read (`kept_fact`)."""

UNCACHED = {"JAX_ENABLE_COMPILATION_CACHE": "false"}
"""A process with JAX's persistent cache off: each program no file hands it compiles, which its compile
telemetry names."""


def kept_worker(directory, mode: str, scale: int = 2, offset: int = 1, environment=None) -> dict:
    """What tests/kept_programs_worker.py prints in a fresh process with JAX's persistent cache off
    (`UNCACHED`): `mode` over the programs file in `directory` at `scale` and `offset`, `kept_module` read
    from `directory`, with `environment`'s variables beside this process's."""
    path = os.pathsep.join((str(directory), str(process_support.REPO_ROOT / "src"),
                            str(process_support.REPO_ROOT / "tests")))
    return process_support.fresh(process_support.REPO_ROOT / "tests" / "kept_programs_worker.py", mode,
                                 str(directory / "programs.dew-aot"), str(scale), str(offset),
                                 cwd=process_support.REPO_ROOT,
                                 variables={"PYTHONPATH": path, **UNCACHED, **(environment or {})})


def kept_in(directory, scale: int = 2, offset: int = 1, module: str = KEPT_MODULE, environment=None) -> dict:
    """What a fresh process keeping its programs in `directory` at `scale` and `offset`, its `kept_module`
    `module`, answers and keeps."""
    directory.mkdir(exist_ok=True)
    (directory / "kept_module.py").write_text(module)
    return kept_worker(directory, "keep", scale, offset, environment)


def test_a_program_one_process_kept_runs_in_another_that_compiles_nothing(tmp_path):
    """A process keeps the executable of each program it compiles at Dew's compile point
    (`dew.training.ahead.keeping`) by its fingerprint and saves them; a later process of the same target
    handed the file (`dew.training.ahead.kept`) traces and lowers each call, finds its executable by the
    program it lowers to, and answers every call exactly as the first did, compiling none of them. Each
    program is its own: a static argument that is None and one that is an empty tuple, and two functions
    written on one line that differ in a constant. A call that differs in what its trace reads lowers
    another program, which the handed process compiles and answers as a process compiling it does: at
    another static argument, and at another value of a global its function names, a function of the user's
    it calls names or code nested in it names, or of a class attribute a static argument's `__call__` reads,
    each of which its module's code changed at run time. A call inside one of Dew's own contexts, and a
    function that reads an array outside its arguments, are kept by their programs as any other; none is
    refused. A CPU's file states its host's processor (row 5265: a warm scoring server paid its programs'
    compiles)."""
    saved = kept_in(tmp_path / "kept")
    answers = saved["answers"]
    assert saved["kept"] == len(answers) + 2 and saved["refused"] == {}
    assert (saved["cpu"] is not None) == (jax.default_backend() == "cpu")
    assert answers["none"] != answers["empty"] and answers["one"] != answers["two"]
    assert kept_worker(tmp_path / "kept", "take") == {"answers": answers, "compiled": []}
    other = kept_worker(tmp_path / "kept", "take", scale=3, offset=4)
    assert other["compiled"] == ["applied", "helped", "nested", "scaled", "shifted"]
    compiled = kept_in(tmp_path / "other", scale=3, offset=4)["answers"]
    assert other["answers"] == compiled
    differing = {name for name in answers if answers[name] != compiled[name]}
    assert differing == {"applied", "helped", "nested", "scaled", "shifted"}


EVERY_PROGRAM = ["<lambda>", "applied", "chosen", "helped", "nested", "scaled", "shifted"]
"""Every function the kept-programs worker calls a program of, by its name."""


@pytest.mark.parametrize(("changed", "compiled"), [
    ("source", ["scaled"]), ("environment", []), ("fact", []), ("setting", []),
    ("flags", EVERY_PROGRAM), ("target", EVERY_PROGRAM)])
def test_programs_kept_under_other_code_settings_or_target_run_where_they_lower_the_same(tmp_path, changed,
                                                                                    compiled):
    """A file of kept programs handed to a process that differs from the one that kept it runs every
    program its calls still lower to, compiles the rest and answers as a process compiling every call
    does, raising nothing: a module's source edited since (the function `scaled` calls `sin` for `tanh`)
    compiles that program alone; an environment variable of Dew's, a fact of the process its code declares
    its traces read (`traced_fact`) and a setting of JAX's (the default PRNG implementation) that none of
    the programs reads lower the same programs, which run as kept; an XLA flag that decides what a program
    computes (deterministic ops) or a host other than this one is another target (`BuildTarget`), whose
    file hands nothing, so every call compiles."""
    from dew.training.ahead import AheadPrograms

    kept_in(tmp_path / "kept")
    flags = f"{os.environ.get('XLA_FLAGS', '')} --xla_gpu_deterministic_ops=true".strip()
    environment = {"environment": {"DEW_KEPT_PROGRAMS_TEST": "1"}, "fact": {"KEPT_FACT": "another"},
                   "setting": {"JAX_DEFAULT_PRNG_IMPL": "rbg"}, "flags": {"XLA_FLAGS": flags}}.get(changed)
    module = KEPT_MODULE
    if changed == "source":
        module = KEPT_MODULE.replace("jnp.tanh(values)", "jnp.sin(values)")
        (tmp_path / "kept" / "kept_module.py").write_text(module)
    elif changed == "target":
        path = tmp_path / "kept" / "programs.dew-aot"
        programs = AheadPrograms.load(path)
        dataclasses.replace(programs, target=dataclasses.replace(programs.target, host="another")).save(path)
    taken = kept_worker(tmp_path / "kept", "take", environment=environment)
    assert taken["compiled"] == compiled
    assert taken["answers"] == kept_in(tmp_path / "fresh", module=module, environment=environment)["answers"]


CHECKED_MODULE = '''
import jax.numpy as jnp
from jax.experimental import checkify


def _doubled(values):
    checkify.check(jnp.all(values > 0), "the values must be positive")
    return values * 2


def checked(values):
    return checkify.checkify(_doubled)(values)
'''
"""A program whose check a run keeps (`checked`): its error tree holds the check's traceback."""

CHECKED_RUN = """
import json, sys
import jax.numpy as jnp
from jax.experimental import checkify
import checked_module
from dew.nn.kernels.options import owned_program
from dew.training.ahead import keeping, kept

mode, path = sys.argv[1], sys.argv[2]
program = owned_program(checked_module.checked)


def refused():
    error, _ = program(-jnp.ones(3))
    try:
        error.throw()
    except checkify.JaxRuntimeError as raised:
        return [type(raised).__name__, str(raised)]
    return None


if mode == "keep":
    with keeping() as programs:
        program(jnp.ones(3))
    programs.save(path)
    print(json.dumps({"refused": refused(), "kept": len(programs.kept)}))
else:
    with kept(path):
        print(json.dumps({"refused": refused()}))
"""
"""A process that keeps a checked program, or is handed it, and reports what its failed check raises."""


class Side(enum.Enum):
    """Two labels of no plain form a program's results may carry."""

    LEFT = "left"
    RIGHT = "right"


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class Labelled:
    """Results and their labels, the labels a node's static data (`Side`), as a tree of results states which
    leaf is which."""

    values: tuple
    labels: tuple

    def tree_flatten(self):
        return self.values, self.labels

    @classmethod
    def tree_unflatten(cls, labels, values):
        return cls(tuple(values), labels)


DIGESTED = """
import json
import jax.numpy as jnp
import checked_module
from dew.nn.kernels.options import owned_program, program_digest

print(json.dumps({"digest": program_digest(owned_program(checked_module.checked).lower(jnp.ones(3)))}))
"""
"""A process printing the digest of the checked program's lowering (`program_digest`)."""


def test_a_programs_digest_is_one_in_every_process_and_its_trees_part_programs(tmp_path):
    """A program's digest (`program_digest`) states its trees node by node: a checkified program, whose error
    tree holds a traceback JAX prints by its address and whose frames name where its code is installed,
    digests the same in two processes of the code installed in two places, as its lowered text is the same,
    its frames stated by module, line and function (`dew._stated.traceback_frames`), the same run as a
    script or under `-c` (its module found under the working directory, train 966's F6), and apart from one
    whose check sits a line further down, which a kept program's failing check would report;
    and two programs of one text whose trees differ (a mapping of other keys, or of one
    enumeration's members labelling its results the other way round) digest apart, since an executable
    takes and gives its leaves by its trees."""
    from dew.nn.kernels.options import owned_program, program_digest

    places = [tmp_path / "one", tmp_path / "other" / "install", tmp_path / "moved"]
    # The third's check a line further down its file.
    texts = [CHECKED_MODULE, CHECKED_MODULE, CHECKED_MODULE.replace("def _doubled", "\ndef _doubled")]
    for place, text in zip(places, texts, strict=True):
        place.mkdir(parents=True)
        (place / "checked_module.py").write_text(text)
    first, second, moved = (process_support.fresh(DIGESTED, cwd=place)["digest"] for place in places)
    # The first's run again as a script beside its module, its path's first entry that directory where
    # under `-c` it is the working directory (`''`).
    (places[0] / "digested.py").write_text(DIGESTED)
    scripted = process_support.fresh(places[0] / "digested.py", cwd=places[0])["digest"]
    assert first == second == scripted != moved
    values = jnp.ones(3)
    one = owned_program(lambda tree: tree["a"] * 2).lower({"a": values})
    other = owned_program(lambda tree: tree["b"] * 2).lower({"b": values})
    assert program_digest(one) != program_digest(other)
    # One text, its results labelled by a node's data of no plain form the other way round: a handed
    # executable gives its results in its own tree, so the labels are part of the program.
    first = owned_program(lambda x: Labelled((x * 2, x * 3), (Side.LEFT, Side.RIGHT))).lower(values)
    swapped = owned_program(lambda x: Labelled((x * 2, x * 3), (Side.RIGHT, Side.LEFT))).lower(values)
    assert program_digest(first) != program_digest(swapped)


def test_a_kept_programs_failed_check_raises_where_handed_as_where_kept(tmp_path):
    """A checked program kept by one process (its error tree's tracebacks held as their frames,
    `HeldTraceback`) and handed to another raises its failed check there as the keeping process does, the
    check's own error and message, not an error of the held traceback's."""
    import process_support

    (tmp_path / "checked_module.py").write_text(CHECKED_MODULE)
    path = str(tmp_path / "checked.dew-aot")
    # The module beside the file, on the path as a user's own.
    found = {"PYTHONPATH": os.pathsep.join((str(tmp_path), str(process_support.REPO_ROOT / "src"),
                                            str(process_support.REPO_ROOT / "tests")))}
    keep = process_support.fresh(CHECKED_RUN, "keep", path, variables=found)
    take = process_support.fresh(CHECKED_RUN, "take", path, variables=found)
    assert keep["kept"] == 1 and keep["refused"] is not None
    assert take["refused"] == keep["refused"]


# Its import is read only as a call of `run` traces.
LAZY_FRONT = "def run(x):\n    import lazy_impl\n\n    return lazy_impl.add(x)\n"
"""An installed module whose function imports another installed module inside its body."""


LAZY_KEPT = """
import json
import site
import sys

site.USER_SITE = sys.argv[3]
sys.path.insert(0, sys.argv[3])
import front
import jax.numpy as jnp
import numpy as np

from dew.nn.kernels.options import owned_program
from dew.telemetry.compiles import Compiles
from dew.training.ahead import keeping, kept

run = owned_program(front.run)
if sys.argv[1] == "keep":
    with keeping() as programs:
        np.asarray(run(jnp.ones(2)))
    programs.save(sys.argv[2])
    print(json.dumps({"kept": len(programs.kept)}))
else:
    with kept(sys.argv[2]), Compiles(None) as compiles:
        answer = np.asarray(run(jnp.ones(2))).tolist()
    ran = [program.name for program in compiles.programs if program.kind == "compiled"]
    print(json.dumps({"answer": answer, "compiled": [name for name in ran if name == "jit(run)"]}))
"""
"""A process keeping `front.run`'s program (`keep`), whose trace imports `lazy_impl`, or handed it from the
kept file (`take`): the call's answer and the compiles it needed."""


def test_a_kept_program_whose_installed_dependency_changed_in_place_compiles_its_own(tmp_path):
    """A kept program of an installed distribution's code, handed in a later process, the dependency it
    calls changed in place since (`x + 1` to `x + 2`), its version and RECORD as they were: the call lowers
    another program than the kept one, finds no executable by its fingerprint, and compiles and runs
    `x + 2`; unchanged, it runs the kept one, compiling nothing."""
    from wheels import installed

    site_packages = tmp_path / "site"
    installed(site_packages, "front", {"front/__init__.py": LAZY_FRONT})
    installed(site_packages, "lazy-impl", {"lazy_impl.py": "def add(x):\n    return x + 1\n"})
    path = tmp_path / "lazy.dew-aot"
    kept = process_support.fresh(LAZY_KEPT, "keep", str(path), str(site_packages), cwd=tmp_path)
    assert kept == {"kept": 1}
    same = process_support.fresh(LAZY_KEPT, "take", str(path), str(site_packages), cwd=tmp_path,
                                 variables=UNCACHED)
    assert same == {"answer": [2.0, 2.0], "compiled": []}
    (site_packages / "lazy_impl.py").write_text("def add(x):\n    return x + 2\n")
    changed = process_support.fresh(LAZY_KEPT, "take", str(path), str(site_packages), cwd=tmp_path,
                                    variables=UNCACHED)
    assert changed == {"answer": [3.0, 3.0], "compiled": ["jit(run)"]}


def test_a_kept_program_is_handed_by_its_own_fingerprint_alone(tmp_path):
    """A file whose programs are kept under fingerprints none of the calls lowers to (here each recorded
    key changed) hands none of them: every call compiles and answers as the keeping process did. A file of
    programs compiled ahead by name, none kept by its program, is refused as it is handed."""
    from dew.training.ahead import AheadPrograms, kept

    answers = kept_in(tmp_path)["answers"]
    path = tmp_path / "programs.dew-aot"
    programs = AheadPrograms.load(path)
    assert programs.kept is not None
    other = {f"{number:064d}": each for number, each in enumerate(programs.kept.values())}
    dataclasses.replace(programs, kept=other).save(path)
    assert kept_worker(tmp_path, "take") == {"answers": answers, "compiled": EVERY_PROGRAM}
    AheadPrograms(programs.target, {}, tuned=True, at_run=()).save(path)
    with pytest.raises(ValueError), kept(path):
        pass


@dataclasses.dataclass(frozen=True, eq=False)
class Factor:
    """A static argument whose field holds a NumPy scalar or a 0-d array of one value."""
    value: np.generic | np.ndarray


def by_factor_kind(held: Factor, values: jax.Array) -> jax.Array:
    """`values` twice where the static argument holds a scalar, three times where it holds an array."""
    return values * (2 if isinstance(held.value, np.generic) else 3)


def test_a_program_kept_over_a_numpy_scalar_is_not_handed_for_a_0d_array_of_its_value(tmp_path):
    """A static argument's record holding `np.float32(1)` and one holding `np.array(1, np.float32)` state
    the same bytes, and a trace may branch on which it holds: a program kept over the scalar is not handed
    to a call over the array, which runs its own trace (main gave the scalar's executable back, 2 * x for
    3 * x)."""
    from dew.nn.kernels.options import owned_program
    from dew.training.ahead import keeping, kept

    values = jnp.ones(3)
    path = tmp_path / "programs.dew-aot"
    fresh = functools.partial(owned_program, by_factor_kind, static_argnums=(0,))
    with keeping() as programs:
        assert fresh()(Factor(np.float32(1)), values).tolist() == [2.0] * 3
    programs.save(path)
    with kept(path):
        assert fresh()(Factor(np.array(1, np.float32)), values).tolist() == [3.0] * 3


def test_a_kept_program_closing_over_an_array_is_handed_to_a_call_closing_over_those_values_alone(tmp_path):
    """A program whose function reads an array outside its arguments is kept with that array a constant of
    its program; handed where a fresh program closes over another array of the same shape and dtype, the
    call runs over the array it closes over, never the kept one's values."""
    from dew.nn.kernels.options import owned_program
    from dew.training.ahead import keeping, kept

    def closing(held: jax.Array):
        return owned_program(lambda values: values + held)

    values = jnp.ones(3)
    path = tmp_path / "programs.dew-aot"
    with keeping() as programs:
        assert closing(jnp.arange(3.0))(values).tolist() == [1.0, 2.0, 3.0]
    programs.save(path)
    assert len(programs.kept) == 1
    with kept(path):
        assert closing(jnp.arange(3.0) * 10)(values).tolist() == [1.0, 11.0, 21.0]
        assert closing(jnp.arange(3.0))(values).tolist() == [1.0, 2.0, 3.0]


@dataclasses.dataclass(frozen=True, eq=False)
class Scales:
    """A static argument whose field holds dictionaries through NumPy's storage of pointers."""
    table: np.ndarray | np.void


def scaled_by_table(held: Scales, values: jax.Array) -> jax.Array:
    """`values` times the scale the static argument's first dictionary holds as the call traces."""
    entry = held.table["entry"] if held.table.dtype.names else held.table[0]
    return values * entry["scale"]


@pytest.mark.parametrize("holder", ["array", "structured"])
def test_a_program_over_numpys_pointers_runs_what_they_point_to_when_its_handed(tmp_path, holder):
    """A static argument holding a dictionary through an array of objects, or a structured scalar with an
    object field, holds the dictionary by its pointer, the same after the dictionary changes: a fresh
    program called where the file is handed after the dictionary changed lowers the scale it holds then,
    another program than the kept one, and runs it, not the executable of the one it held before."""
    from dew.nn.kernels.options import owned_program
    from dew.training.ahead import keeping, kept

    dictionary = {"scale": 2}
    table = np.empty(1, dtype=object)
    table[0] = dictionary
    if holder == "structured":
        table = np.array([(dictionary,)], dtype=[("entry", object)])[0]
    values = jnp.ones(3)
    path = tmp_path / "programs.dew-aot"
    # A fresh program at each call, as a fresh process makes it.
    fresh = functools.partial(owned_program, scaled_by_table, static_argnums=(0,))
    with keeping() as programs:
        assert fresh()(Scales(table), values).tolist() == [2.0] * 3
    programs.save(path)
    dictionary["scale"] = 3
    with kept(path):
        assert fresh()(Scales(table), values).tolist() == [3.0] * 3




def test_a_keeping_block_within_another_is_heard_by_both_and_the_outer_alone_after_it():
    """Each executable Dew's compile point compiles within a `keeping` block nested in another is heard by
    both, the outer first: a save of a run's programs inside a save of its process's keeps it in each (row
    8450: the inner block hid the outer, whose save then lacked it); past the inner block, the outer
    alone hears."""
    from dew.nn.kernels.options import keeping, owned_program

    heard: list[tuple[str, str]] = []

    def listener(name):
        return lambda program, lowered, executable: heard.append((name, program.called))

    def doubled(values):
        return values * 2

    def halved(values):
        return values / 2

    with keeping(listener("outer")):
        with keeping(listener("inner")):
            owned_program(doubled)(jnp.arange(3.))
        owned_program(halved)(jnp.arange(3.))
    assert [name for name, _ in heard] == ["outer", "inner", "outer"], heard
    assert heard[0][1] == heard[1][1] != heard[2][1], heard


def test_a_save_of_programs_within_another_keeps_its_program_in_both_and_the_later_one_in_the_outer():
    """Row 8450 through the saves a run makes (`dew.training.ahead.keeping`): a program compiled within a
    save nested in another is kept by both, and one compiled past the inner save by the outer alone, each
    kept by its fingerprint, so each file hands back what was compiled while it was open."""
    from dew.nn.kernels.options import owned_program
    from dew.training.ahead import keeping

    def doubled(values):
        return values * 2

    def halved(values):
        return values / 2

    with keeping() as outer:
        with keeping() as inner:
            owned_program(doubled)(jnp.arange(3.))
        owned_program(halved)(jnp.arange(3.))
    assert not inner.refused and not outer.refused, (inner.refused, outer.refused)
    assert len(inner.kept) == 1 and len(outer.kept) == 2 and set(inner.kept) < set(outer.kept), (
        inner.kept, outer.kept)


def _doubled(x):
    """A module's function a program compiles (the identity witness's)."""
    return 2 * x


def _halved(x):
    """Another module's function (the identity witness's)."""
    return x / 2


def test_a_program_dews_compile_point_compiles_is_stated_by_its_function_and_arguments():
    """A program Dew's compile point compiles (`dew.jit`'s, an `OuterProgram`) is stated by its function and
    its jit arguments (`dew._stated.stated`), so a check-1 arm of one holds a statement and prepares (rows
    10096 on: dew.jit arms ran unprepared): two programs of one function under one set of arguments state
    alike, one called (its executables cached) as one never called, and another function or another
    argument states another identity."""
    from dew._stated import stated
    from dew.nn.kernels.options import owned_program

    called = owned_program(_doubled)
    called(np.ones(3, np.float32))
    one = stated(owned_program(_doubled), "a program")
    assert stated(called, "a program called") == one
    assert stated(owned_program(_halved), "another function's") != one
    assert stated(owned_program(_doubled, donate_argnums=(0,)), "another argument's") != one


_FACTOR = 2.0
"""A global a module's function reads (the library-function witness's)."""


def _by_factor(x):
    """A module's function of the user's reading a global of its module (the library-function witness's)."""
    return x * _FACTOR


def test_a_librarys_local_function_is_stated_by_its_code_and_closure_not_its_modules_globals(monkeypatch):
    """A library's local function (`jax.checkpoint`'s, which Dew's SwiGLU holds) is stated by its module, its
    name, its code, its closure and its defaults, its globals its module's at its installed version
    (`dew._stated.reads`): Dew's SwiGLU states, alike made twice (rows 10064 on: its arms ran unprepared);
    two over one function of the user's state alike and over two apart; and a function of the user's own
    code is still stated with its globals, one changed another identity."""
    from dew._stated import stated
    from dew.nn import activations

    assert stated(activations.gated_product("swiglu"), "a SwiGLU") == stated(
        activations.gated_product("swiglu"), "a SwiGLU made again")
    assert stated(jax.checkpoint(_doubled), "a checkpoint") == stated(jax.checkpoint(_doubled), "again")
    assert stated(jax.checkpoint(_doubled), "a checkpoint") != stated(jax.checkpoint(_halved), "another's")
    before = stated(_by_factor, "a user's function")
    monkeypatch.setitem(globals(), "_FACTOR", 3.0)
    assert stated(_by_factor, "its global changed") != before


def _identity(x, y):
    """A function a custom derivative wraps (the custom-derivative witness's)."""
    return x + y


def _doubled_tangent(primals, tangents):
    return _identity(*primals), 2 * tangents[0] + tangents[1]


def _tripled_tangent(primals, tangents):
    return _identity(*primals), 3 * tangents[0] + tangents[1]


def _with_jvp(rule):
    made = jax.custom_jvp(_identity)
    made.defjvp(rule)
    return made


def _with_jvps(scale):
    """A custom JVP of one rule an argument (`defjvps`, a closure over them), the second's scaled."""
    made = jax.custom_jvp(_identity)
    made.defjvps(lambda tangent, out, x, y: tangent, lambda tangent, out, x, y: scale * tangent)
    return made


def _forward(x, y):
    return _identity(x, y), None


def _backward(_, cotangent):
    return cotangent, cotangent


def test_a_custom_derivative_is_stated_by_its_function_and_its_rules():
    """A function with a custom derivative (`jax.custom_jvp`, `jax.custom_vjp`) is stated by its function,
    its rules and the options a trace reads (`dew._stated.stated`), so an arm holding a test's own rule has a
    statement (row 10087: it ran unprepared): one made twice of the same functions states alike, two whose
    rule differs apart, two whose per-argument rules (`defjvps`) differ in one argument's apart, and a custom
    VJP before its rules are defined states as such, apart from one defined."""
    from dew._stated import stated

    assert stated(_with_jvp(_doubled_tangent), "a rule") == stated(_with_jvp(_doubled_tangent), "again")
    assert stated(_with_jvp(_doubled_tangent), "a rule") != stated(_with_jvp(_tripled_tangent), "another")
    assert stated(_with_jvps(2.0), "per-argument rules") != stated(_with_jvps(3.0), "one argument's apart")
    undefined, defined = jax.custom_vjp(_identity), jax.custom_vjp(_identity)
    defined.defvjp(_forward, _backward)
    assert stated(undefined, "no rules yet") != stated(defined, "its rules")


def test_a_librarys_sentinel_in_its_closure_is_stated_by_its_cell():
    """A library's function closing over its module's sentinel (`jax.vmap`'s axis name when none is given, a
    bare `object()`) is stated by that cell's name (`dew._stated.reads`): the batched form of one function
    states alike made twice and apart from another's or one naming its axis (row 10169's batched arms had no
    statement and ran unprepared)."""
    from dew._stated import stated

    assert stated(jax.vmap(_doubled), "a batch") == stated(jax.vmap(_doubled), "a batch made again")
    assert stated(jax.vmap(_doubled), "a batch") != stated(jax.vmap(_halved), "another's batch")
    assert stated(jax.vmap(_doubled), "a batch") != stated(jax.vmap(_doubled, axis_name="i"), "a named axis")
