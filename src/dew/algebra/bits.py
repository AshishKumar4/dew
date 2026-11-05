"""Bit-level forms (docs/design/algebra.md §7): a declared transform of a stored word into the registers a
kernel's arithmetic reads, a low-precision format's decode among them, and its cheapest instruction form over
a GPU's integer pipes, found by an exhaustive search ordered by price and proven exact on every word.

A transform is GF(2)-linear: each bit a consumer reads is the parity of some of the input's bits, so a
format's decode, its codes placed where the consumer's multiply reads them, is one (`BitTransform.placed`).
Where its layout is free, the stored word may be any invertible re-encoding of the input, chosen with the
form: its encoder runs once where the weights are stored, the decoder each time they are read."""

from __future__ import annotations

import dataclasses
import enum
import itertools
from collections.abc import Iterator, Mapping, Sequence
from typing import NamedTuple

import numpy as np

WORD = 32
"""A register's bits: every value a form computes is one 32-bit register."""

MASK = (1 << WORD) - 1


class Unit(enum.Enum):
    """The multiprocessor's pipe an instruction issues on (Nsight Compute's Kernel Profiling Guide, its
    pipelines): the ALU executes logic, shifts and every integer instruction but IMAD and IMUL, which the FMA
    pipe executes. Each takes 64 results a clock per multiprocessor of a 32-bit shift, bitwise operation and
    integer multiply-add alike, on every compute capability from 7.5 to 12.1 (the CUDA C++ Best Practices
    Guide 13, Table 5), so a form's price is its busiest unit's issues."""

    ALU = "alu"
    FMA = "fma"


class Op(enum.Enum):
    """An instruction of the vocabulary, each a GF(2)-linear map of its operands' words, as SASS names it, and
    the pipes that can issue it (`units`). A shift's amount and a mask are immediates."""

    SHR = "shf.r.u32"      # a logical right shift: SHF on the ALU, or IMAD.HI.U32 by 2^(32-c) on the FMA pipe
    ROTR = "shf.r.w.u32"   # a right rotation: SHF's funnel shift of the word with itself, the ALU's alone
    SHL = "shl"            # a left shift: SHF.L on the ALU, or IMAD.SHL (a multiply by 2^k) on the FMA pipe
    LOP3 = "lop3"          # a mask of the XOR of one or two values: LOP3, its mask an immediate

    @property
    def units(self) -> tuple[Unit, ...]:
        return (Unit.ALU,) if self in (Op.ROTR, Op.LOP3) else (Unit.ALU, Unit.FMA)


class Step(NamedTuple):
    """One instruction of a form: `op` over the values at `args` (0 the stored word, `i + 1` step `i`'s), its
    immediate (a shift's amount, a mask), and the unit it issues on."""

    op: Op
    args: tuple[int, ...]
    immediate: int
    unit: Unit


def _applied(step: Step, values: Sequence[np.ndarray]) -> np.ndarray:
    """`step` over `values` (uint64 arrays of 32-bit words), as the instruction computes it."""
    first = values[step.args[0]]
    if step.op is Op.SHR:
        return first >> np.uint64(step.immediate)
    if step.op is Op.ROTR:
        count = np.uint64(step.immediate)
        return ((first >> count) | (first << np.uint64(WORD - step.immediate))) & np.uint64(MASK)
    if step.op is Op.SHL:
        return (first << np.uint64(step.immediate)) & np.uint64(MASK)
    xor = first if len(step.args) == 1 else first ^ values[step.args[1]]
    return xor & np.uint64(step.immediate)


def _bits(mask: int) -> list[int]:
    """The set bits of `mask`, lowest first."""
    return [bit for bit in range(WORD) if mask >> bit & 1]


def _parities(words: np.ndarray, rows: Sequence[int]) -> np.ndarray:
    """Each of `words` mapped by `rows` (its bit `j` the parity of the word's bits in `rows[j]`)."""
    out = np.zeros_like(words)
    for index, row in enumerate(rows):
        bit = np.bitwise_count(words & np.uint64(row)) & np.uint64(1)
        out |= bit << np.uint64(index)
    return out


def _inverse(rows: Sequence[int]) -> tuple[int, ...] | None:
    """The inverse of the square GF(2) map `rows` (output bit `j` the parity of the input's bits in
    `rows[j]`), in the same form, or None where it is singular."""
    size = len(rows)
    augmented = [(row, 1 << index) for index, row in enumerate(rows)]
    for column in range(size):
        pivot = next((index for index in range(column, size) if augmented[index][0] >> column & 1), None)
        if pivot is None:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        row, image = augmented[column]
        for index in range(size):
            if index != column and augmented[index][0] >> column & 1:
                augmented[index] = (augmented[index][0] ^ row, augmented[index][1] ^ image)
    # Row `j` now reads input bit `j` alone, its image the output bits whose parity it is: the inverse's row.
    return tuple(image for _, image in augmented)


def _composed(outer: Sequence[int], inner: Sequence[int]) -> tuple[int, ...]:
    """`outer` after `inner`, both GF(2) maps as rows."""
    return tuple(_parity_of(outer_row, inner) for outer_row in outer)


def _parity_of(row: int, inner: Sequence[int]) -> int:
    """The row reading, through `inner`, the parity of `inner`'s outputs at `row`'s bits."""
    out = 0
    for bit in _bits(row):
        out ^= inner[bit]
    return out


@dataclasses.dataclass(frozen=True)
class BitTransform:
    """A declared GF(2)-linear transform of a `width`-bit input into registers, register `i`'s bits at
    `outputs[i]` (a mask: the bits its consumer reads, every other bit zero), each the parity of the input's
    bits its row names (`rows`: register by register, each mask's bits lowest first). `free`: the stored word
    may be any invertible re-encoding of the input (its encoder run where the weights are stored), so a form
    need only read its bits back in some order; else the stored word is the input."""

    name: str
    width: int
    outputs: tuple[int, ...]
    rows: tuple[int, ...]
    free: bool = True

    def __post_init__(self):
        if not 0 < self.width <= WORD or len(self.rows) != sum(len(_bits(mask)) for mask in self.outputs):
            raise ValueError(f"{self.name}: a row for each of its outputs' bits, over at most {WORD} bits")
        if any(row >> self.width for row in self.rows):
            raise ValueError(f"{self.name}: a row reads a bit past its {self.width}-bit input")
        if self.free and (len(self.rows) != self.width or _inverse(self.rows) is None):
            raise ValueError(f"{self.name}: a free layout reads each of its input's bits back exactly once")

    @classmethod
    def placed(cls, name: str, width: int, outputs: Sequence[int], placement: Mapping[int, tuple[int, int]],
               *, free: bool = True) -> BitTransform:
        """A bit placement: input bit `j` read at register `placement[j][0]`'s bit `placement[j][1]`, a
        format's decode, its codes' fields placed where the consumer's arithmetic reads them."""
        source = {target: bit for bit, target in placement.items()}
        rows = tuple(1 << source[register, bit] if (register, bit) in source else 0
                     for register, mask in enumerate(outputs) for bit in _bits(mask))
        return cls(name, width, tuple(outputs), rows, free)

    def applied(self, inputs: np.ndarray) -> tuple[np.ndarray, ...]:
        """Each register the transform gives each of `inputs`."""
        values, rows = [], iter(self.rows)
        for mask in self.outputs:
            out = np.zeros_like(inputs)
            for bit in _bits(mask):
                out |= (np.bitwise_count(inputs & np.uint64(next(rows))) & np.uint64(1)) << np.uint64(bit)
            values.append(out)
        return tuple(values)


@dataclasses.dataclass(frozen=True)
class Form:
    """A transform's instruction form: its `steps`, the value each output register is (`outputs`, 0 the stored
    word, `i + 1` step `i`'s), and the stored word's encoder (`encoder`: its bit `j` the parity of the input's
    bits in `encoder[j]`; the identity for a fixed layout)."""

    steps: tuple[Step, ...]
    outputs: tuple[int, ...]
    encoder: tuple[int, ...]

    def decoded(self, words: np.ndarray) -> tuple[np.ndarray, ...]:
        """Each output register of each stored word of `words`."""
        values = [words.astype(np.uint64)]
        for step in self.steps:
            values.append(_applied(step, values))
        return tuple(values[index] for index in self.outputs)

    def encoded(self, inputs: np.ndarray) -> np.ndarray:
        """The stored word of each of `inputs`."""
        return _parities(inputs.astype(np.uint64), self.encoder)

    @property
    def issues(self) -> dict[Unit, int]:
        """Its instructions on each unit."""
        return {unit: sum(step.unit is unit for step in self.steps) for unit in Unit}

    @property
    def depth(self) -> int:
        """Its dependent instructions' longest chain."""
        depths = [0]
        for step in self.steps:
            depths.append(1 + max(depths[arg] for arg in step.args))
        return max(depths[index] for index in self.outputs)

    @property
    def price(self) -> tuple[int, int]:
        """Its busiest unit's issues, then its instructions (`Unit`: one rate): a decoder's words are many
        and independent, so its pipes' throughput prices it, not its chain's latency."""
        return max(self.issues.values()), len(self.steps)


class Unproven(ValueError):
    """A form that does not compute its transform on some word."""


def proven(transform: BitTransform, form: Form) -> None:
    """Prove `form` computes `transform` on every input, or raise `Unproven`: every instruction of the
    vocabulary is GF(2)-linear (`Op`), so the form and its encoder are linear maps, and two linear maps that
    agree on each basis word agree on all 2^width; the input's every word is checked besides where it has 16
    bits or fewer. Bits outside a register's mask are proven zero by the same argument."""
    if len(form.encoder) != transform.width:
        raise Unproven(f"{transform.name}: an encoder of {len(form.encoder)} bits for a "
                       f"{transform.width}-bit input")
    inputs = (np.arange(1 << transform.width, dtype=np.uint64) if transform.width <= 16
              else np.array([1 << bit for bit in range(transform.width)], dtype=np.uint64))
    got = form.decoded(form.encoded(inputs))
    for register, (value, expected) in enumerate(zip(got, transform.applied(inputs), strict=True)):
        wrong = np.flatnonzero(value != expected)
        if wrong.size:
            raise Unproven(f"{transform.name}: register {register} of input {int(inputs[wrong[0]]):#x} is "
                           f"{int(value[wrong[0]]):#x}, not {int(expected[wrong[0]]):#x}")


class _Shift(NamedTuple):
    """A value a form's registers read besides the stored word: a right shift or rotation of it, or a left
    shift of it or of the right one (`base`: 0 the stored word, 1 the right shift), by `amount`; and the input
    bit each of its bits is (`bits`, -1 a zero)."""

    op: Op
    base: int
    amount: int
    bits: tuple[int, ...]


def _stored(width: int) -> tuple[int, ...]:
    """The input bit each bit of a `width`-bit stored word is in its register (-1 a zero past it)."""
    return tuple(t if t < width else -1 for t in range(WORD))


class _Shape(NamedTuple):
    """A form's shifts: its right-moving source's (`right`, None for none) and its count of left shifts."""

    right: Op | None
    lefts: int


def _shifted(bits: Sequence[int], op: Op, amount: int) -> tuple[int, ...]:
    """The input bit each bit of a value is, the value `bits` shifted by `op`."""
    if op is Op.SHR:
        return tuple(bits[t + amount] if t + amount < WORD else -1 for t in range(WORD))
    if op is Op.ROTR:
        return tuple(bits[(t + amount) % WORD] for t in range(WORD))
    return tuple(bits[t - amount] if t >= amount else -1 for t in range(WORD))


def _priced(shape: _Shape, registers: int) -> tuple[tuple[int, int], int]:
    """A shape's price (`Form.price`) with one LOP3 a register, and its shifts on the ALU: a rotation there
    (the ALU's alone), the rest split so the two pipes balance."""
    shifts = (shape.right is not None) + shape.lefts
    rotations = int(shape.right is Op.ROTR)
    on_alu = min(shifts, max(rotations, -(-(shifts - registers) // 2)))
    return (max(registers + on_alu, shifts - on_alu), registers + shifts), on_alu


def _configs(shape: _Shape, reach: int, width: int) -> Iterator[list[_Shift]]:
    """Every shift set of `shape` over a `width`-bit stored word: its right shift by each amount (first), and
    each choice of left shifts of the stored word or the right shift by at most `reach`."""
    stored = _stored(width)
    rights = ([None] if shape.right is None
              else [_Shift(shape.right, 0, amount, _shifted(stored, shape.right, amount))
                    for amount in range(1, WORD)])
    for right in rights:
        bases = [stored] if right is None else [stored, right.bits]
        lefted = [_Shift(Op.SHL, index, amount, _shifted(bits, Op.SHL, amount))
                  for index, bits in enumerate(bases) for amount in range(1, reach + 1)]
        for chosen in itertools.combinations(lefted, shape.lefts):
            yield [*([] if right is None else [right]), *chosen]


def _covered(transform: BitTransform, shifts: Sequence[_Shift]) -> bool:
    """Whether the stored word and `shifts` can read every input bit back where the registers read (a free
    layout's necessary condition): a register's bit is a parity of the sources' bits at its place, so each
    place holds at most as many independent rows as registers read it and as distinct bits the sources hold
    there."""
    readers: dict[int, int] = {}
    for mask in transform.outputs:
        for t in _bits(mask):
            readers[t] = readers.get(t, 0) + 1
    sources = (_stored(transform.width), *(shift.bits for shift in shifts))
    held = {t: {bits[t] for bits in sources} - {-1} for t in readers}
    reached = set().union(*held.values())
    return (len(reached) >= transform.width
            and sum(min(count, len(held[t])) for t, count in readers.items()) >= transform.width)


def _independent(basis: dict[int, int], rows: Sequence[int]) -> dict[int, int] | None:
    """`basis` (pivot bit to row, each row free of the others' pivots) grown by `rows`, or None where one of
    them is spanned by the rest."""
    grown = dict(basis)
    for row in rows:
        for pivot, pivot_row in grown.items():
            if row >> pivot & 1:
                row ^= pivot_row
        if not row:
            return None
        pivot = row.bit_length() - 1
        for other, other_row in list(grown.items()):
            if other_row >> pivot & 1:
                grown[other] = other_row ^ row
        grown[pivot] = row
    return grown


def _registers(transform: BitTransform, shifts: Sequence[_Shift]) -> Iterator[list[tuple[int, int]]]:
    """Each assignment of a LOP3 a register, `mask & (u ^ v)` or `mask & u` (a pair `(u, u)`) over the
    sources (0 the stored word, `i + 1` shift `i`) and the registers before it, under which the form reads the
    transform back: its registers' rows the transform's own (a fixed layout) or independent (a free one), and
    every shift read, directly or through a shift or a register reading it (else a cheaper set reads the
    same)."""
    masks = [_bits(mask) for mask in transform.outputs]
    starts = list(itertools.accumulate((len(bits) for bits in masks), initial=0))
    wanted = [transform.rows[starts[i]:starts[i + 1]] for i in range(len(masks))]
    sources = [_stored(transform.width), *(shift.bits for shift in shifts)]
    count = len(sources)
    bases = {shift.base for shift in shifts if shift.op is Op.SHL and shift.base}
    needed = frozenset(range(1, count)) - bases

    def search(index: int, built: list[list[int]], chosen: list[tuple[int, int]], basis: dict[int, int],
               read: frozenset[int]) -> Iterator[list[tuple[int, int]]]:
        if index == len(masks):
            if read >= needed:
                yield list(chosen)
            return
        place = masks[index]
        # A register reads an earlier one at its own mask's bits: the earlier one's rows where the masks meet.
        rows_of = [[1 << bits[t] if bits[t] >= 0 else 0 for t in place] for bits in sources]
        rows_of += [[built[j][masks[j].index(t)] if t in masks[j] else 0 for t in place]
                    for j in range(index)]
        # Registers of one mask that read sources alone are interchangeable: their pairs in increasing order.
        floor = (chosen[-1] if transform.free and index and masks[index] == masks[index - 1]
                 and chosen[-1][1] < count else (0, 0))
        for first in range(count + index):
            for second in range(first, count + index):
                if (second == first and first >= count) or (second < count and (first, second) < floor):
                    continue   # a register copied, or a reordering of one already tried
                rows = (rows_of[first] if second == first
                        else [a ^ b for a, b in zip(rows_of[first], rows_of[second], strict=True)])
                if transform.free:
                    grown = _independent(basis, rows)
                    if grown is None:
                        continue
                else:
                    if tuple(rows) != wanted[index]:
                        continue
                    grown = basis
                reads = read | {arg for arg in (first, second) if arg < count}
                yield from search(index + 1, [*built, rows], [*chosen, (first, second)], grown, reads)

    yield from search(0, [], [], {}, frozenset())


def _form(transform: BitTransform, shifts: Sequence[_Shift], on_alu: int,
          registers: Sequence[tuple[int, int]]) -> Form:
    """The form of `shifts` (each a step, `on_alu` of them on the ALU, a rotation first) and `registers`, its
    encoder solved from its map (a free layout's) or the identity."""
    alu = set(sorted(range(len(shifts)), key=lambda index: shifts[index].op is not Op.ROTR)[:on_alu])
    steps = [Step(shift.op, (shift.base,), shift.amount, Unit.ALU if index in alu else Unit.FMA)
             for index, shift in enumerate(shifts)]
    sources = len(shifts) + 1
    outputs: list[int] = []
    for register, (first, second) in enumerate(registers):
        ids = tuple(arg if arg < sources else outputs[arg - sources]
                    for arg in ((first,) if first == second else (first, second)))
        steps.append(Step(Op.LOP3, ids, transform.outputs[register], Unit.ALU))
        outputs.append(len(steps))
    plain = Form(tuple(steps), tuple(outputs), tuple(1 << bit for bit in range(transform.width)))
    if not transform.free:
        return plain
    # The form's map from the stored word to the bits the registers hold, inverted: the encoder is that
    # inverse after the transform.
    basis = np.array([1 << bit for bit in range(transform.width)], dtype=np.uint64)
    decoded = plain.decoded(basis)
    read = tuple(sum(((int(decoded[register][bit]) >> t) & 1) << bit for bit in range(transform.width))
                 for register, mask in enumerate(transform.outputs) for t in _bits(mask))
    inverse = _inverse(read)
    if inverse is None:
        raise Unproven(f"{transform.name}: a free form's registers read its stored bits back twice")
    return dataclasses.replace(plain, encoder=_composed(inverse, transform.rows))


def synthesized(transform: BitTransform, *, right: bool = True, lefts: int = 3,
                reach: int = WORD - 1) -> Form:
    """The cheapest proven form of `transform` (`Form.price`) of one LOP3 a register over the stored word, at
    most one right shift or rotation of it (`right`) and `lefts` left shifts of it or of that, each by at most
    `reach` places, each shift on the pipe that balances the two: the search walks shapes in increasing price
    and returns the first form found, proven (`proven`). Raises LookupError where no form within the bounds
    computes it."""
    registers = len(transform.outputs)
    shapes = [_Shape(op, count) for op in (None, *((Op.SHR, Op.ROTR) if right else ()))
              for count in range(lefts + 1)]
    for shape in sorted(shapes, key=lambda shape: _priced(shape, registers)[0]):
        on_alu = _priced(shape, registers)[1]
        for shifts in _configs(shape, reach, transform.width):
            if transform.free and not _covered(transform, shifts):
                continue
            for assignment in itertools.islice(_registers(transform, shifts), 1):
                form = _form(transform, shifts, on_alu, assignment)
                proven(transform, form)
                return form
    raise LookupError(f"{transform.name}: no form of {lefts} left shifts within {reach} places computes it")
