# /// script
# dependencies = ["z3-solver==5.3.3.0"]
# ///
"""Search a cheaper instruction form of a bit transform that is not linear (docs/design/algebra.md §7.3, row
4050): its reference, a jax.numpy function of uint32 words, traced (`seed`) into a straight-line form of
`dew.algebra.bits`'s vocabulary, then each window of up to three steps that only its last step leaves
replaced by a cheaper program of one or two steps that computes the same on every input word, found by
counterexample-guided synthesis over Z3's bit-vectors (`improved`): a program's operations, its operands,
its immediates and its truth tables all unknowns, a candidate that matches the examples checked for every
word, a word it differs on added. A window is replaced only where Z3 proves the replacement equal on all
2^32 values of each of its inputs; the result is proven again against the reference on every word by its
consumer's test. Run at development time (Z3 is this script's dependency, not Dew's):

    uv run tools/superoptimize_bits.py e4m3
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax
import jax.extend
import jax.numpy as jnp
import numpy as np
import z3

from dew.algebra import bits
from dew.algebra.bits import BitForm, BitOp, BitStep, Unit

MASK = bits.MASK

# --- the reference traced into a form ---------------------------------------------------------------------

_PREDICATES = {"lt": 0, "le": 1, "gt": 2, "ge": 3, "eq": 4, "ne": 5}
_FLIPPED = {"lt": "gt", "le": "ge", "gt": "lt", "ge": "le", "eq": "eq", "ne": "ne"}


class _Seeded:
    """A form being written from a jaxpr: its steps, each jaxpr value a place (`_Place`, a step's value) or a
    constant (a Python int, a scalar's)."""

    def __init__(self):
        self.steps: list[BitStep] = []
        self.made: dict[BitStep, int] = {}

    def step(self, op: BitOp, args, immediate: int = 0, table: int = 0) -> int:
        """A step's value, one step for each distinct instruction (a reference's repeated expression once)."""
        unit = Unit.FMA if op is BitOp.IMAD else Unit.ALU
        made = BitStep(op, tuple(args), int(immediate) & MASK, unit, table)
        if made not in self.made:
            self.steps.append(made)
            self.made[made] = len(self.steps)
        return self.made[made]

    def held(self, value) -> int:
        """A value as a step's: a constant materialized (an IADD3 of none)."""
        return value if isinstance(value, _Place) else _Place(self.step(BitOp.IADD3, (), int(value)))


class _Place(int):
    """A value a step (or the stored word, 0) holds, apart from a constant."""


def seed(reference, *, signed_inputs: bool = False) -> BitForm:
    """The straight-line form of `reference` (a jax.numpy function of a uint32 array, its result uint32),
    each integer primitive of its jaxpr one step of the vocabulary: shifts, logic, adds, minima, compares and
    selects; dtype conversions between 32-bit integers and booleans (0 or 1) none."""
    closed = jax.make_jaxpr(reference)(jax.ShapeDtypeStruct((8,), jnp.uint32))
    form = _Seeded()
    env: dict = {}

    def read(var):
        if isinstance(var, jax.extend.core.Literal):
            return int(np.asarray(var.val).astype(np.int64)) & MASK
        return env[var]

    def run(closed, inputs):
        jaxpr, consts = (closed.jaxpr, closed.consts) if hasattr(closed, "consts") else (closed, ())
        env.update(zip(jaxpr.invars, inputs, strict=True))
        env.update({var: int(np.asarray(const).astype(np.int64)) & MASK
                    for var, const in zip(jaxpr.constvars, consts, strict=True)})
        for equation in jaxpr.eqns:
            args = [read(var) for var in equation.invars]
            name = equation.primitive.name
            signed = any(getattr(var.aval, "dtype", None) == jnp.int32 for var in equation.invars)
            if name in ("pjit", "jit", "closed_call"):
                env.update(zip(equation.outvars, run(equation.params["jaxpr"], args), strict=True))
                continue
            env[equation.outvars[0]] = _seeded(form, name, args, signed, equation)
        return [read(var) for var in jaxpr.outvars]

    [result] = run(closed, [_Place(0)])
    return BitForm(tuple(form.steps), (form.held(result),), tuple(1 << bit for bit in range(bits.WORD)))


def _seeded(form: _Seeded, name: str, args, signed: bool, equation):
    if name in ("convert_element_type", "broadcast_in_dim", "copy"):
        return args[0]
    constant = [not isinstance(arg, _Place) for arg in args]
    if all(constant):
        raise ValueError(f"{name}: a step of constants alone (fold it in the reference)")
    if name in ("shift_right_logical", "shift_left", "shift_right_arithmetic"):
        value, places = form.held(args[0]), args[1]
        if not isinstance(places, _Place):
            op = {"shift_right_logical": BitOp.SHR, "shift_left": BitOp.SHL,
                  "shift_right_arithmetic": BitOp.SAR}[name]
            return _Place(form.step(op, (value,), places)) if places else value
        return _Place(form.step(BitOp.SHRV if name == "shift_right_logical" else BitOp.SHLV, (value, places)))
    if name in ("and", "or", "xor"):
        table = {"and": 0xF0 & 0xCC, "or": 0xF0 | 0xCC, "xor": 0xF0 ^ 0xCC}[name]
        places = [arg for arg in args if isinstance(arg, _Place)]
        constants = [arg for arg in args if not isinstance(arg, _Place)]
        return _Place(form.step(BitOp.LOGIC, places, constants[0] if constants else 0, table))
    if name == "not":
        return _Place(form.step(BitOp.LOGIC, (args[0],), 0, 0x0F))
    if name == "add":
        places = [arg for arg in args if isinstance(arg, _Place)]
        return _Place(form.step(BitOp.IADD3, places, sum(arg for arg in args if not isinstance(arg, _Place))))
    if name == "sub":
        left, right = args
        if constant[0]:
            return _Place(form.step(BitOp.IADD3, (right,), left, 0b1))
        if constant[1]:
            return _Place(form.step(BitOp.IADD3, (left,), -right))
        return _Place(form.step(BitOp.IADD3, (left, right), 0, 0b10))
    if name in ("min", "max"):
        places = [arg for arg in args if isinstance(arg, _Place)]
        constants = [arg for arg in args if not isinstance(arg, _Place)]
        return _Place(form.step(BitOp.MINMAX, places, constants[0] if constants else 0,
                                int(signed) | (2 if name == "max" else 0)))
    if name in _PREDICATES:
        predicate = name if not constant[0] else _FLIPPED[name]
        places = [arg for arg in args if isinstance(arg, _Place)]
        constants = [arg for arg in args if not isinstance(arg, _Place)]
        return _Place(form.step(BitOp.CMP, places, constants[0] if constants else 0,
                                _PREDICATES[predicate] | (8 if signed else 0)))
    if name == "select_n":
        predicate, false, true = args
        return _Place(form.step(BitOp.SEL, (form.held(predicate), form.held(true), form.held(false))))
    raise ValueError(f"{name}: no step of the vocabulary")


# --- each step's meaning over Z3's bit-vectors --------------------------------------------------------------

def _z3_step(op: BitOp, operands: list, immediate, table) -> z3.BitVecRef:
    """`op`'s value over bit-vector `operands`, its `immediate` and `table` bit-vectors (`bits._run`'s)."""
    zero, word = z3.BitVecVal(0, 32), z3.BitVecVal(bits.WORD, 32)
    first = operands[0] if operands else zero
    second = operands[1] if len(operands) > 1 else immediate
    if op is BitOp.LOGIC:
        values = [*operands, immediate, immediate][:3]
        value = zero
        for minterm in range(8):
            term = z3.BitVecVal(MASK, 32)
            for index, operand in enumerate(values):
                term = term & (operand if minterm >> (2 - index) & 1 else ~operand)
            bit = z3.Extract(minterm, minterm, table)
            value = value | z3.If(bit == 1, term, zero)
        return value
    if op is BitOp.IADD3:
        total = immediate
        for index, operand in enumerate(operands):
            total = total + z3.If(z3.Extract(index, index, table) == 1, -operand, operand)
        return total
    if op is BitOp.IMAD:
        return first * immediate + (operands[1] if len(operands) > 1 else zero)
    if op is BitOp.MINMAX:
        signed, greatest = z3.Extract(0, 0, table) == 1, z3.Extract(1, 1, table) == 1
        less = z3.If(signed, first < second, z3.ULT(first, second))
        return z3.If(greatest, z3.If(less, second, first), z3.If(less, first, second))
    if op is BitOp.SHR:
        return z3.LShR(first, immediate)
    if op is BitOp.SHL:
        return first << immediate
    if op is BitOp.SAR:
        return first >> immediate
    if op in (BitOp.SHRV, BitOp.SHLV):
        shifted = z3.LShR(first, second) if op is BitOp.SHRV else first << second
        return z3.If(z3.ULT(second, word), shifted, zero)
    if op is BitOp.CMP:
        signed = z3.Extract(3, 3, table) == 1
        code = z3.Extract(2, 0, table)
        lt = z3.If(signed, first < second, z3.ULT(first, second))
        le = z3.If(signed, first <= second, z3.ULE(first, second))
        holds = z3.If(code == 0, lt, z3.If(code == 1, le, z3.If(code == 2, z3.Not(le), z3.If(
            code == 3, z3.Not(lt), z3.If(code == 4, first == second, first != second)))))
        return z3.If(holds, z3.BitVecVal(1, 32), zero)
    if op is BitOp.SEL:
        return z3.If(first != zero, operands[1], operands[2])
    if op is BitOp.PRMT:
        low, high = operands[0], operands[-1]
        sources = [z3.Extract(8 * (k % 4) + 7, 8 * (k % 4), low if k < 4 else high) for k in range(8)]
        made = []
        for k in range(4):
            nibble = z3.Extract(4 * k + 2, 4 * k, immediate)
            byte = sources[7]
            for index in range(6, -1, -1):
                byte = z3.If(nibble == index, sources[index], byte)
            made.append(byte)
        return z3.Concat(*reversed(made))
    raise ValueError(f"{op}: not in the synthesis vocabulary")


ARITIES = {BitOp.LOGIC: (1, 2, 3), BitOp.IADD3: (1, 2), BitOp.IMAD: (1, 2), BitOp.MINMAX: (1, 2),
           BitOp.SHR: (1,), BitOp.SHL: (1,), BitOp.SAR: (1,), BitOp.SHRV: (2,), BitOp.SHLV: (2,),
           BitOp.CMP: (1, 2), BitOp.SEL: (3,), BitOp.PRMT: (1, 2)}
"""The ops a replacement may take and the operand counts each reads (the immediate the rest)."""


# --- windows and their replacements -------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Window:
    """Steps `first`..`last` (1-based values) of a form whose only value read past them is the last's, and
    the values outside it they read (`inputs`)."""

    first: int
    last: int
    inputs: tuple[int, ...]


def windows(form: BitForm, longest: int = 3) -> list[Window]:
    readers: dict[int, set[int]] = {}
    for index, step in enumerate(form.steps, start=1):
        for arg in step.args:
            readers.setdefault(arg, set()).add(index)
    for output in form.outputs:
        readers.setdefault(output, set()).add(len(form.steps) + 1)
    found = []
    for last in range(1, len(form.steps) + 1):
        for first in range(max(1, last - longest + 1), last + 1):
            inside = set(range(first, last + 1))
            if any(readers.get(value, set()) - inside for value in inside - {last}):
                continue
            inputs = sorted({arg for value in inside for arg in form.steps[value - 1].args} - inside)
            if len(inputs) <= 3 and sum(form.steps[value - 1].op.issues for value in inside) >= 2:
                found.append(Window(first, last, tuple(inputs)))
    return found


def _window_value(form: BitForm, window: Window, inputs: list) -> z3.BitVecRef:
    values = dict(zip(window.inputs, inputs, strict=True))
    for index in range(window.first, window.last + 1):
        step = form.steps[index - 1]
        values[index] = _z3_step(step.op, [values[arg] for arg in step.args],
                                 z3.BitVecVal(step.immediate, 32), z3.BitVecVal(step.table, 32))
    return values[window.last]


def _program(shape: tuple[tuple[BitOp, int], ...], inputs: list, tag: str):
    """A program of `shape` (each step's op and operand count) over `inputs`, its operand choices, immediates
    and tables unknowns: its value and its unknowns."""
    values = list(inputs)
    unknowns = []
    for index, (op, arity) in enumerate(shape):
        immediate = z3.BitVec(f"{tag}i{index}", 32)
        table = z3.BitVec(f"{tag}t{index}", 32)
        choices = [z3.BitVec(f"{tag}c{index}_{slot}", 8) for slot in range(arity)]
        operands = [_chosen(choice, values) for choice in choices]
        values.append(_z3_step(op, operands, immediate, table))
        unknowns.append((immediate, table, choices, len(values) - 1))
    return values[-1], unknowns


def _chosen(choice, values):
    picked = values[-1]
    for index in range(len(values) - 2, -1, -1):
        picked = z3.If(choice == index, values[index], picked)
    return picked


def _bounded(unknowns, available: int):
    """The unknowns' constraints: each operand choice one of the values before its step."""
    return [z3.ULT(choice, z3.BitVecVal(place, 8)) for _, _, choices, place in unknowns for choice in choices]


_TABLES = {BitOp.LOGIC: 0xFF, BitOp.IADD3: 0b111, BitOp.MINMAX: 0b11, BitOp.CMP: 0b1111}
"""The bits of `table` each op reads (none for the rest)."""

_SHIFTS = (BitOp.SHR, BitOp.SHL, BitOp.SAR)
"""The ops whose immediate is a shift's amount, below the word's width."""


def replaced(form: BitForm, window: Window, budget: int, seconds: float, longest: int = 2
             ) -> list[BitStep] | None:
    """A program of fewer issues than `budget` computing the window's value from its inputs on every word,
    or None: shapes of one step, then two, up to `longest`, each by counterexample-guided synthesis."""
    names = [f"x{index}" for index in range(len(window.inputs))]
    symbols = [z3.BitVec(name, 32) for name in names]
    target = _window_value(form, window, symbols)
    ops = list(ARITIES)
    shapes = [((op, arity),) for op in ops for arity in ARITIES[op]]
    singles = [(op, arity) for op in ops for arity in ARITIES[op]]
    for length in range(2, longest + 1):
        shapes += list(itertools.product(singles, repeat=length))
    started = time.perf_counter()
    for shape in shapes:
        if sum(op.issues for op, _ in shape) >= budget:
            continue
        if time.perf_counter() - started > seconds:
            return None
        found = _cegis(shape, symbols, target, len(window.inputs))
        if found is not None:
            return _steps(shape, found, window)
    return None


def _cegis(shape, symbols, target, inputs: int):
    examples: list[list[int]] = [[0] * inputs, [MASK] * inputs, [0x7F800000] * inputs, [0x3F800000] * inputs]
    rng = np.random.default_rng(4050)
    examples += [list(map(int, rng.integers(0, 1 << 32, inputs, dtype=np.uint64))) for _ in range(4)]
    for _ in range(64):
        solver = z3.Solver()
        solver.set("timeout", 2000)
        program_unknowns = None
        for example in examples:
            values = [z3.BitVecVal(value, 32) for value in example]
            program, unknowns = _program(shape, values, "")
            program_unknowns = unknowns
            expected = z3.substitute(target, *zip(symbols, values, strict=True))
            solver.add(program == expected)
        solver.add(*_bounded(program_unknowns, inputs))
        for (op, _), (immediate, _, _, _) in zip(shape, program_unknowns, strict=True):
            if op in _SHIFTS:
                solver.add(z3.ULT(immediate, z3.BitVecVal(bits.WORD, 32)))
            if op is BitOp.PRMT:
                # Literal byte selection alone: no nibble's sign-replicating bit, no bits past four nibbles.
                solver.add(immediate & z3.BitVecVal(0xFFFF8888, 32) == 0)
        if solver.check() != z3.sat:
            return None
        model = solver.model()
        concrete = [(model.eval(immediate, model_completion=True).as_long(),
                     model.eval(table, model_completion=True).as_long(),
                     [model.eval(choice, model_completion=True).as_long() for choice in choices])
                    for immediate, table, choices, _ in program_unknowns]
        checked = z3.Solver()
        checked.set("timeout", 10000)
        program = _fixed(shape, concrete, symbols)
        checked.add(program != target)
        verdict = checked.check()
        if verdict == z3.unsat:
            return concrete
        if verdict != z3.sat:
            return None
        witness = checked.model()
        examples.append([witness.eval(symbol, model_completion=True).as_long() for symbol in symbols])
    return None


def _fixed(shape, concrete, inputs):
    values = list(inputs)
    for (op, _), (immediate, table, choices) in zip(shape, concrete, strict=True):
        values.append(_z3_step(op, [values[choice] for choice in choices], z3.BitVecVal(immediate, 32),
                               z3.BitVecVal(table, 32)))
    return values[-1]


def _steps(shape, concrete, window: Window) -> list[BitStep]:
    """The replacement's steps, operand `i` of the program's values the window's input `i` or the program's
    own step, numbered from the window's first place."""
    places = list(window.inputs)
    steps = []
    for (op, _), (immediate, table, choices) in zip(shape, concrete, strict=True):
        unit = Unit.FMA if op is BitOp.IMAD else Unit.ALU
        steps.append(BitStep(op, tuple(places[choice] for choice in choices), immediate, unit,
                             table & _TABLES.get(op, 0)))
        places.append(window.first + len(steps) - 1)
    return steps


def spliced(form: BitForm, window: Window, steps: list[BitStep]) -> BitForm:
    """`form` with the window's steps replaced by `steps`, every later reference renumbered."""
    removed = window.last - window.first + 1
    shift = removed - len(steps)

    def renumbered(arg: int) -> int:
        if arg < window.first:
            return arg
        if arg == window.last:
            return window.first + len(steps) - 1
        return arg - shift

    tail = [step._replace(args=tuple(renumbered(arg) for arg in step.args))
            for step in form.steps[window.last:]]
    return BitForm((*form.steps[:window.first - 1], *steps, *tail),
                   tuple(renumbered(output) for output in form.outputs), form.encoder)


def _searched(job):
    form, window, seconds, longest = job
    budget = sum(form.steps[index - 1].op.issues for index in range(window.first, window.last + 1))
    steps = replaced(form, window, budget, seconds, longest)
    return window, (budget - sum(step.op.issues for step in steps)) if steps else 0, steps


def improved(form: BitForm, seconds: float = 60.0, widest: int = 3, longest: int = 2) -> BitForm:
    """`form` with windows replaced until none of up to three steps has a cheaper equal program: each round
    searches every window at once (a process each core), then splices the savings that touch no other,
    largest first, from the form's end so earlier places stand."""
    with concurrent.futures.ProcessPoolExecutor() as pool:
        while True:
            jobs = [(form, place, seconds, longest) for place in windows(form, widest)]
            found = [result for result in pool.map(_searched, jobs) if result[1]]
            if not found:
                return form
            taken: list = []
            for window, saving, steps in sorted(found, key=lambda result: -result[1]):
                if all(window.last < other.first or other.last < window.first for other, _, _ in taken):
                    taken.append((window, saving, steps))
            for window, saving, steps in sorted(taken, key=lambda result: -result[0].first):
                print(f"K window {window.first}..{window.last} saves {saving}: "
                      f"{[(step.op.name, step.args, hex(step.immediate), step.table) for step in steps]}",
                      flush=True)
                form = spliced(form, window, steps)
            print(f"K round: {len(form.steps)} steps, price {form.price}", flush=True)


def _halved(words, minifloat: bits.Minifloat, register: int):
    """Register `register` of a word of four FP8 codes in their natural order (code `i` at bits `8 i`): codes
    `2 register` and `2 register + 1` as its low and high bfloat16 halves, each the code's value times
    2^(minifloat.bias - 127), its sign at the half's top and its other bits under the half's exponent and
    mantissa (a finite code's placement, the free layout's `synthesized` decode in the stored order)."""
    rest = minifloat.width - 1

    def half(code):
        return (code & jnp.uint32(1 << rest)) << jnp.uint32(15 - rest) | \
            (code & jnp.uint32((1 << rest) - 1)) << jnp.uint32(7 - minifloat.mantissa)

    first, second = (words >> jnp.uint32(16 * register + 8 * k) & jnp.uint32(0xFF) for k in (0, 1))
    return half(first) | half(second) << jnp.uint32(16)


def _e8m0(words):
    """An MX block scale (OCP Microscaling Formats v1.0, 5.4.1: E8M0, its low byte) as fp32 bits: 2^(e-127),
    0x00 the subnormal 2^-127, 0xFF a NaN."""
    e = words & jnp.uint32(0xFF)
    return jnp.where(e == 0, jnp.uint32(0x00400000), jnp.where(e == 0xFF, jnp.uint32(0x7FC00000),
                                                                e << jnp.uint32(23)))


REFERENCES = {"e4m3": lambda words: bits.encoded(words, bits.E4M3),
              "e5m2": lambda words: bits.encoded(words, bits.E5M2),
              "e2m1": lambda words: bits.encoded(words, bits.E2M1),
              **{f"{name}-decode-{register}": (lambda words, m=minifloat, r=register: _halved(words, m, r))
                 for name, minifloat in (("e4m3", bits.E4M3), ("e5m2", bits.E5M2)) for register in (0, 1)},
              "e8m0": _e8m0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("name", choices=sorted(REFERENCES))
    parser.add_argument("--seconds", type=float, default=60.0, help="each window's search bound")
    parser.add_argument("--window", type=int, help="search the seed's window of this index alone")
    parser.add_argument("--longest", type=int, default=3, help="a window's most steps")
    parser.add_argument("--steps", type=int, default=2, help="a replacement's most steps")
    args = parser.parse_args()
    reference = REFERENCES[args.name]
    form = seed(reference)
    if args.window is not None:
        found = windows(form)
        if args.window >= len(found):
            return
        window = found[args.window]
        budget = sum(form.steps[index - 1].op.issues for index in range(window.first, window.last + 1))
        began = time.perf_counter()
        steps = replaced(form, window, budget, args.seconds)
        print(f"K window {args.window} {window} budget {budget} {time.perf_counter() - began:.0f}s -> "
              f"{None if steps is None else [(s.op.name, s.args, hex(s.immediate), s.table) for s in steps]}",
              flush=True)
        return
    words = np.random.default_rng(0).integers(0, 1 << 32, 1 << 20, dtype=np.uint64).astype(np.uint32)
    edges = np.array([0, 1, 0x7F800000, 0x7FC00000, 0xFF800000, 0x80000000, 0x7F7FFFFF, MASK], np.uint32)
    words = np.concatenate([words, edges])
    want = np.asarray(reference(jnp.asarray(words)))
    assert (form.decoded(words)[0] == want).all(), "the seed is not its reference"
    print(f"K seed: {len(form.steps)} steps, price {form.price}, windows {len(windows(form))}", flush=True)
    for index, step in enumerate(form.steps, start=1):
        print(f"K step {index}: {step.op.name} {step.args} {step.immediate:#x} {step.table}", flush=True)
    better = improved(form, args.seconds, args.longest, args.steps)
    assert (better.decoded(words)[0] == want).all(), "the improved form is not its reference"
    print(f"K improved: {len(better.steps)} steps, price {better.price}", flush=True)
    for index, step in enumerate(better.steps, start=1):
        print(f"K final {index}: {step.op.name} {step.args} {step.immediate:#x} {step.table} "
              f"{step.unit.name}", flush=True)
    print(f"K outputs {better.outputs}", flush=True)


if __name__ == "__main__":
    main()
