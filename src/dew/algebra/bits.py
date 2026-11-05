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

import jax
import jax.numpy as jnp
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


class BitOp(enum.Enum):
    """An instruction of the vocabulary, each a GF(2)-linear map of its operands' words, as SASS names it, and
    the pipes that can issue it (`units`). A shift's amount and a mask are immediates."""

    SHR = "shf.r.u32"      # a logical right shift: SHF on the ALU, or IMAD.HI.U32 by 2^(32-c) on the FMA pipe
    ROTR = "shf.r.w.u32"   # a right rotation: SHF's funnel shift of the word with itself, the ALU's alone
    SHL = "shl"            # a left shift: SHF.L on the ALU, or IMAD.SHL (a multiply by 2^k) on the FMA pipe
    LOP3 = "lop3"          # a mask of the XOR of one or two values: LOP3, its mask an immediate

    @property
    def units(self) -> tuple[Unit, ...]:
        return (Unit.ALU,) if self in (BitOp.ROTR, BitOp.LOP3) else (Unit.ALU, Unit.FMA)


class BitStep(NamedTuple):
    """One instruction of a form: `op` over the values at `args` (0 the stored word, `i + 1` step `i`'s), its
    immediate (a shift's amount, a mask), and the unit it issues on."""

    op: BitOp
    args: tuple[int, ...]
    immediate: int
    unit: Unit


def _run[Words: (np.ndarray, jax.Array)](steps: Sequence[BitStep], outputs: Sequence[int], words: Words,
                                          xp) -> tuple[Words, ...]:
    """Each output (`outputs`, 0 the stored word, `i + 1` step `i`'s) of `steps` over `words`, uint32 arrays
    of `xp` (NumPy, or `jax.numpy` for a kernel's portable path): the one evaluation of a form's steps, as
    each instruction computes it, its shifts in 32 bits."""
    values = [words]
    for step in steps:
        first, amount = values[step.args[0]], xp.uint32(step.immediate)
        if step.op is BitOp.SHR:
            value = first >> amount
        elif step.op is BitOp.ROTR:
            value = (first >> amount) | (first << xp.uint32(WORD - step.immediate))
        elif step.op is BitOp.SHL:
            value = first << amount
        else:
            value = (first if len(step.args) == 1 else first ^ values[step.args[1]]) & amount
        values.append(value)
    return tuple(values[index] for index in outputs)


def _bits(mask: int) -> list[int]:
    """The set bits of `mask`, lowest first."""
    return [bit for bit in range(WORD) if mask >> bit & 1]


def _parities[Words: (np.ndarray, jax.Array)](words: Words, rows: Sequence[int], xp=np) -> Words:
    """Each of `words`, unsigned integers of `xp` (NumPy, or `jax.numpy` for a kernel's prepare step), mapped
    by `rows` (its bit `j` the parity of the word's bits in `rows[j]`): the one evaluation of an encoder."""
    word = words.dtype.type
    out = xp.zeros_like(words)
    for index, row in enumerate(rows):
        bit = xp.bitwise_count(words & word(row)).astype(words.dtype) & word(1)
        out |= bit << word(index)
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
    need only read its bits back in some order; else the stored word is the input. `units`: the pipes its
    form may issue on (the ALU always, which issues every LOP3; a consumer whose own math holds the FMA pipe
    keeps its decode off it)."""

    name: str
    width: int
    outputs: tuple[int, ...]
    rows: tuple[int, ...]
    free: bool = True
    units: frozenset[Unit] = frozenset(Unit)

    def __post_init__(self):
        if not 0 < self.width <= WORD or len(self.rows) != sum(len(_bits(mask)) for mask in self.outputs):
            raise ValueError(f"{self.name}: a row for each of its outputs' bits, over at most {WORD} bits")
        if any(row >> self.width for row in self.rows):
            raise ValueError(f"{self.name}: a row reads a bit past its {self.width}-bit input")
        if self.free and (len(self.rows) != self.width or _inverse(self.rows) is None):
            raise ValueError(f"{self.name}: a free layout reads each of its input's bits back exactly once")
        if Unit.ALU not in self.units:
            raise ValueError(f"{self.name}: every form masks on the ALU, which its units leave out")

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
class BitForm:
    """A transform's instruction form: its `steps`, the value each output register is (`outputs`, 0 the stored
    word, `i + 1` step `i`'s), and the stored word's encoder (`encoder`: its bit `j` the parity of the input's
    bits in `encoder[j]`; the identity for a fixed layout)."""

    steps: tuple[BitStep, ...]
    outputs: tuple[int, ...]
    encoder: tuple[int, ...]

    def decoded(self, words: np.ndarray) -> tuple[np.ndarray, ...]:
        """Each output register of each stored word of `words` (`_run`'s, in NumPy), as uint64."""
        return tuple(value.astype(np.uint64)
                     for value in _run(self.steps, self.outputs, words.astype(np.uint32), np))

    def encoded(self, inputs: np.ndarray) -> np.ndarray:
        """The stored word of each of `inputs`, as uint64."""
        return _parities(inputs.astype(np.uint64), self.encoder)

    def prepared(self, inputs: jax.Array) -> jax.Array:
        """The stored word of each of `inputs` (uint32), as JAX computes it (`_parities`'s, in `jax.numpy`):
        a consumer's prepare step, which stores its weights in the form's layout once."""
        return _parities(inputs.astype(jnp.uint32), self.encoder, jnp)

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

    def traced(self, words: jax.Array) -> tuple[jax.Array, ...]:
        """Each output register of each stored word of `words` (uint32), as JAX computes it (`_run`'s, in
        `jax.numpy`): a kernel's portable path and its interpreter's, where no inline assembly runs."""
        return _run(self.steps, self.outputs, words.astype(jnp.uint32), jnp)

    def values(self, words: jax.Array, dtype) -> jax.Array:
        """The `dtype` elements the registers of each stored word of `words` (uint32, `[..., c]`) hold
        (`traced`), register by register and each register's low bits first: `[..., c * n]`, `n` the
        registers' elements a word (a decode's elements in their natural order, `NVFP4_BFLOAT16`'s eight)."""
        registers = jnp.stack(self.traced(words), axis=-1)
        return jax.lax.bitcast_convert_type(registers, dtype).reshape(*words.shape[:-1], -1)

    def ptx(self, output: int) -> str:
        """Output register `output` of the form as one PTX block over its stored word `$1` into `$0`, the
        steps it reads each on the pipe its step names (`_PTX`), for a kernel's inline assembly (Pallas's
        `elementwise_inline_asm`, constraints `=r,r`, which takes one result a block): a register's block
        apiece, each holding the steps it reads, so a step two registers share is written twice and ptxas,
        which optimizes the kernel's PTX whole, may compute it once (the consumer's compile report shows
        whether it does)."""
        needed: set[int] = set()
        pending = [self.outputs[output]]
        while pending:
            value = pending.pop()
            if value and value not in needed:
                needed.add(value)
                pending.extend(self.steps[value - 1].args)
        names = {0: "$1", **{value: f"v{value}" for value in needed}}
        lines = [f".reg .b32 {', '.join(names[value] for value in sorted(needed))};"] if needed else []
        for value in sorted(needed):
            step = self.steps[value - 1]
            lines.append(_PTX[step.op, step.unit](names[value], [names[arg] for arg in step.args],
                                                  step.immediate))
        lines.append(f"mov.b32 $0, {names[self.outputs[output]]};")
        return "{\n" + "\n".join(lines) + "\n}"

    @property
    def price(self) -> tuple[int, int]:
        """Its busiest unit's issues, then its instructions (`Unit`: one rate): a decoder's words are many
        and independent, so its pipes' throughput prices it, not its chain's latency."""
        return max(self.issues.values()), len(self.steps)


# LOP3's truth table of `c & (a ^ b)` and of `c & a`, over its operands' tables 0xF0, 0xCC and 0xAA (PTX ISA,
# "Logic and Shift Instructions: lop3").
_MASKED_XOR, _MASKED = (0xF0 ^ 0xCC) & 0xAA, 0xF0 & 0xAA

_PTX = {
    (BitOp.SHR, Unit.ALU): lambda out, ins, amount: f"shr.b32 {out}, {ins[0]}, {amount};",
    # IMAD.HI.U32: the high word of a multiply by 2^(32-c) is the logical shift right by c.
    (BitOp.SHR, Unit.FMA): lambda out, ins, amount: f"mul.hi.u32 {out}, {ins[0]}, {1 << (WORD - amount):#x};",
    (BitOp.ROTR, Unit.ALU): lambda out, ins, amount: f"shf.r.wrap.b32 {out}, {ins[0]}, {ins[0]}, {amount};",
    (BitOp.SHL, Unit.ALU): lambda out, ins, amount: f"shl.b32 {out}, {ins[0]}, {amount};",
    # IMAD.SHL: a multiply by 2^k, the FMA pipe's left shift.
    (BitOp.SHL, Unit.FMA): lambda out, ins, amount: f"mul.lo.u32 {out}, {ins[0]}, {1 << amount:#x};",
    (BitOp.LOP3, Unit.ALU): lambda out, ins, mask: (
        f"lop3.b32 {out}, {ins[0]}, {ins[1]}, {mask:#010x}, {_MASKED_XOR:#04x};" if len(ins) == 2
        else f"lop3.b32 {out}, {ins[0]}, {ins[0]}, {mask:#010x}, {_MASKED:#04x};"),
}
"""Each instruction as the PTX written for its pipe: a shift or rotation as `shr`/`shl`/`shf`, which ptxas
issues as SHF on the ALU, and a multiply by a power of two as `mul`, which it issues as IMAD on the FMA pipe
(PTX ISA, "Integer Arithmetic Instructions"; Nsight Compute's pipelines). PTX fixes the arithmetic, not the
SASS: ptxas chooses the final instructions, which the consumer's compile report shows."""


class Unproven(ValueError):
    """A form that does not compute its transform on some word."""


def proven(transform: BitTransform, form: BitForm) -> None:
    """Prove `form` computes `transform` on every input, or raise `Unproven`: every instruction of the
    vocabulary is GF(2)-linear (`BitOp`), so the form and its encoder are linear maps, and two linear maps
    that agree on each basis word agree on all 2^width; the input's every word is checked besides where it
    has 16 bits or fewer. Bits outside a register's mask are proven zero by the same argument. A fixed
    layout's encoder must be the identity: its form reads the input as stored."""
    identity = tuple(1 << bit for bit in range(transform.width))
    if len(form.encoder) != transform.width or (not transform.free and form.encoder != identity):
        raise Unproven(f"{transform.name}: its form reads a stored word its layout does not store")
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

    op: BitOp
    base: int
    amount: int
    bits: tuple[int, ...]


def _stored(width: int) -> tuple[int, ...]:
    """The input bit each bit of a `width`-bit stored word is in its register (-1 a zero past it)."""
    return tuple(t if t < width else -1 for t in range(WORD))


class _Shape(NamedTuple):
    """A form's shifts: its right-moving source's (`right`, None for none) and its count of left shifts."""

    right: BitOp | None
    lefts: int


def _shifted(bits: Sequence[int], op: BitOp, amount: int) -> tuple[int, ...]:
    """The input bit each bit of a value is, the value `bits` shifted by `op`."""
    if op is BitOp.SHR:
        return tuple(bits[t + amount] if t + amount < WORD else -1 for t in range(WORD))
    if op is BitOp.ROTR:
        return tuple(bits[(t + amount) % WORD] for t in range(WORD))
    return tuple(bits[t - amount] if t >= amount else -1 for t in range(WORD))


def _priced(shape: _Shape, registers: int, units: frozenset[Unit]) -> tuple[tuple[int, int], int]:
    """A shape's price (`BitForm.price`) with one LOP3 a register, and its shifts on the ALU: a rotation there
    (the ALU's alone), the rest split so the two pipes balance, or all of them where `units` holds no FMA."""
    shifts = (shape.right is not None) + shape.lefts
    rotations = int(shape.right is BitOp.ROTR)
    on_alu = (shifts if Unit.FMA not in units
              else min(shifts, max(rotations, -(-(shifts - registers) // 2))))
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
        lefted = [_Shift(BitOp.SHL, index, amount, _shifted(bits, BitOp.SHL, amount))
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
    transform back: its registers' rows the transform's own (a fixed layout's, its mask narrowed to the bits
    its transform does not hold zero, `_narrowed`) or independent (a free one's), and every shift read,
    directly or through a shift or a register reading it (else a cheaper set reads the same)."""
    masks = [_bits(mask) for mask in transform.outputs]
    wanted = _wanted(transform)
    sources = [_stored(transform.width), *(shift.bits for shift in shifts)]
    count = len(sources)
    bases = {shift.base for shift in shifts if shift.op is BitOp.SHL and shift.base}
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
                    rows = [row if want else 0 for row, want in zip(rows, wanted[index], strict=True)]
                    if tuple(rows) != wanted[index]:
                        continue
                    grown = basis
                reads = read | {arg for arg in (first, second) if arg < count}
                yield from search(index + 1, [*built, rows], [*chosen, (first, second)], grown, reads)

    yield from search(0, [], [], {}, frozenset())


def _wanted(transform: BitTransform) -> list[tuple[int, ...]]:
    """The transform's rows, register by register."""
    starts = list(itertools.accumulate((len(_bits(mask)) for mask in transform.outputs), initial=0))
    return [transform.rows[starts[i]:starts[i + 1]] for i in range(len(transform.outputs))]


def _narrowed(transform: BitTransform, register: int) -> int:
    """Register `register`'s LOP3 mask: its output mask, a fixed layout's without the bits its transform
    holds zero, which the mask zeroes as it masks the rest."""
    mask = transform.outputs[register]
    if transform.free:
        return mask
    return sum(1 << t for t, row in zip(_bits(mask), _wanted(transform)[register], strict=True) if row)


def _form(transform: BitTransform, shifts: Sequence[_Shift], on_alu: int,
          registers: Sequence[tuple[int, int]]) -> BitForm:
    """The form of `shifts` (each a step, `on_alu` of them on the ALU, a rotation first) and `registers`, its
    encoder solved from its map (a free layout's) or the identity."""
    alu = set(sorted(range(len(shifts)), key=lambda index: shifts[index].op is not BitOp.ROTR)[:on_alu])
    steps = [BitStep(shift.op, (shift.base,), shift.amount, Unit.ALU if index in alu else Unit.FMA)
             for index, shift in enumerate(shifts)]
    sources = len(shifts) + 1
    outputs: list[int] = []
    for register, (first, second) in enumerate(registers):
        ids = tuple(arg if arg < sources else outputs[arg - sources]
                    for arg in ((first,) if first == second else (first, second)))
        steps.append(BitStep(BitOp.LOP3, ids, _narrowed(transform, register), Unit.ALU))
        outputs.append(len(steps))
    plain = BitForm(tuple(steps), tuple(outputs), tuple(1 << bit for bit in range(transform.width)))
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
                reach: int = WORD - 1) -> BitForm:
    """The cheapest proven form of `transform` (`BitForm.price`) of one LOP3 a register over the stored word,
    at most one right shift or rotation of it by any amount (`right`) and `lefts` left shifts of it or of
    that, each by at most `reach` places, each shift on the pipe that balances the two among the transform's
    `units`: the search walks shapes in increasing price and returns the first form found, proven (`proven`).
    Raises LookupError where no form within the bounds computes it."""
    registers, units = len(transform.outputs), transform.units
    shapes = [_Shape(op, count) for op in (None, *((BitOp.SHR, BitOp.ROTR) if right else ()))
              for count in range(lefts + 1)]
    for shape in sorted(shapes, key=lambda shape: _priced(shape, registers, units)[0]):
        on_alu = _priced(shape, registers, units)[1]
        for shifts in _configs(shape, reach, transform.width):
            if transform.free and not _covered(transform, shifts):
                continue
            for assignment in itertools.islice(_registers(transform, shifts), 1):
                form = _form(transform, shifts, on_alu, assignment)
                proven(transform, form)
                return form
    raise LookupError(f"{transform.name}: no form of {lefts} left shifts within {reach} places computes it")


# E2M1 (OCP Microscaling Formats v1.0, 5.3.3): a mantissa bit, two exponent bits of bias 1, a sign; each
# field's bit in a bfloat16 half, so the half times 2^126 is the code's value (a zero exponent a subnormal of
# both).
_E2M1_IN_BFLOAT16 = {0: 6, 1: 7, 2: 8, 3: 15}

NVFP4_BFLOAT16 = BitTransform.placed(
    "nvfp4-bf16", WORD, (0x81C081C0,) * 4,
    {4 * code + field: (code // 2, 16 * (code % 2) + bit)
     for code in range(8) for field, bit in _E2M1_IN_BFLOAT16.items()})
"""Eight E2M1 codes of a word in their natural order (code `i` at bits `4i` to `4i + 3`: a byte's low nibble
the earlier element), decoded into four registers of two bfloat16s, code `2i` the low half of register `i` and
`2i + 1` the high, each the code's value times 2^-126: the exponent's bias, which the block scale's multiply
takes (the GEMM template's matrix-operand hook, algebra.md 7.4). Its layout is free."""

NVFP4_BFLOAT16_FORM = BitForm(
    steps=(BitStep(BitOp.ROTR, (0,), 6, Unit.ALU), BitStep(BitOp.SHL, (0,), 2, Unit.FMA),
           BitStep(BitOp.SHL, (0,), 6, Unit.FMA), BitStep(BitOp.SHL, (1,), 2, Unit.FMA),
           *(BitStep(BitOp.LOP3, args, 0x81C081C0, Unit.ALU) for args in ((0, 1), (0, 3), (0, 4), (2,)))),
    outputs=(5, 6, 7, 8),
    encoder=(0x4000100, 0x8000202, 0x4040401, 0x2800080, 0x1000000, 0x2000000, 0x4000000, 0x8000002,
             0x4040001, 0x20000808, 0x4010000, 0x8020002, 0x4000001, 0x8000000, 0x4040005, 0x20000008,
             0x40001000, 0x80002020, 0x40404010, 0x20080008, 0x10000000, 0x20000000, 0x40000000, 0x80000020,
             0x40400010, 0x2008080, 0x40100000, 0x80200020, 0x40000010, 0x80000000, 0x40400050, 0x2000080))
"""`NVFP4_BFLOAT16`'s form as `synthesized(NVFP4_BFLOAT16, reach=6)` finds it, checked in so a kernel reads it
without searching: a rotation by 6 and left shifts by 2 and 6 and of the rotation's by 2 (on the FMA pipe),
four masked XORs, the fp4 challenge's price (5 issues on the busiest pipe, 8 instructions). Its encoder is
the stored layout's version: weights stored in it decode by this form alone."""


NVFP4_OWED = 2.0 ** 8
"""The factor an NVFP4 tile's decode (`nvfp4_decoded`) owes its values, which the epilogue's fp32 scale takes:
a GPU's prologue multiplies the form's decode (`code 2^-126`, `NVFP4_BFLOAT16`) by its block scale times
2^118, each then in bfloat16's range, the product exact, so the tile is the values over 2^8."""


def nvfp4_decoded(words: jax.Array, scales: jax.Array) -> jax.Array:
    """An NVFP4 tile in bfloat16, each value its E2M1 code times its block's E4M3 scale over `NVFP4_OWED`,
    exactly: `words` `[depth / 8, columns]` uint32, each the stored (`BitForm.prepared`) word of a column's
    eight consecutive codes along the depth, and `scales` `[depth / 16, columns]` float8_e4m3fn, one for
    each sixteen; `[depth, columns]`: the code restored from its decode (`_restored`) times its scale over
    2^8 in float32, exact (at most six significant bits) and within bfloat16's normal range, as a GPU's
    prologue forms it in one multiply per value (`NVFP4_OWED`), a GEMM operand prologue's per stage (the
    template's matrix-operand hook, algebra.md 7.4)."""
    codes = _restored(NVFP4_BFLOAT16_FORM.values(words.T, jnp.bfloat16).T)
    owed = scales.astype(jnp.float32) * np.float32(1 / NVFP4_OWED)
    return (codes * jnp.repeat(owed, 16, axis=0)).astype(jnp.bfloat16)


def _restored(decoded: jax.Array) -> jax.Array:
    """A decode's bfloat16 values (`NVFP4_BFLOAT16_FORM.values`, each `code 2^-126`) times 2^126, in float32,
    by integer arithmetic: a subnormal (a code of 0 or 0.5) as its mantissa's integer times 2^-7, every other
    value's exponent raised by 126, so no lane that flushes subnormals (XLA:CPU's) zeroes a code of 0.5."""
    bits = jax.lax.bitcast_convert_type(decoded, jnp.uint16).astype(jnp.uint32)
    raised = jax.lax.bitcast_convert_type(((bits + (126 << 7)) & 0xFFFF).astype(jnp.uint16), jnp.bfloat16)
    mantissa = (bits & 0x7F).astype(jnp.float32) * np.float32(2.0 ** -7)
    subnormal = jnp.where(bits & 0x8000 != 0, -mantissa, mantissa)
    return jnp.where(bits & 0x7F80 == 0, subnormal, raised.astype(jnp.float32))
