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
    """An instruction of the vocabulary, as SASS names it, and the pipes that can issue it (`units`). The
    first four are GF(2)-linear maps of their operands' words (`linear`), a shift's amount and a mask
    immediates: the linear search's (`synthesized`) and proof's (`proven`). The rest are the integer pipes'
    arithmetic a form that is not linear takes (an encode's rounding and saturation), proven against its
    reference on every word (row 4050); their immediate an operand where they take one."""

    SHR = "shf.r.u32"      # a logical right shift: SHF on the ALU, or IMAD.HI.U32 by 2^(32-c) on the FMA pipe
    ROTR = "shf.r.w.u32"   # a right rotation: SHF's funnel shift of the word with itself, the ALU's alone
    SHL = "shl"            # a left shift: SHF.L on the ALU, or IMAD.SHL (a multiply by 2^k) on the FMA pipe
    LOP3 = "lop3"          # a mask of the XOR of one or two values: LOP3, its mask an immediate
    LOGIC = "lop3.lut"     # LOP3 of any truth table (`table`) over its values, the immediate the last operand
    IADD3 = "iadd3"        # the sum of its values and the immediate, those `table` names negated
    IMAD = "imad"          # its first value times the immediate, plus its second where it has one
    MINMAX = "imnmx"       # the least of its two values, or of its value and the immediate (`table` 1 signed,
                           # 2 the greatest)
    SAR = "shf.r.s32"      # its value shifted right by the immediate, its sign bit filling
    SHRV = "shf.r.u32.v"   # its first value shifted right by its second, 0 past 31 places (PTX clamps at 32)
    SHLV = "shf.l.u32.v"   # its first value shifted left by its second, 0 past 31 places
    CMP = "isetp"          # 1 where its first value compares to its second (or the immediate) as `table`
                           # names (`COMPARISONS`), else 0: ISETP and a SEL of it, two issues
    SEL = "sel"            # its second value where its first is not 0, else its third: ISETP and SEL, two
    PRMT = "prmt"          # each byte one of its two values' eight, the immediate's nibbles naming them
                           # (0-7: literal selection; a nibble's sign-replicating bit is not taken)

    @property
    def units(self) -> tuple[Unit, ...]:
        if self is BitOp.IMAD:
            return (Unit.FMA,)
        return (Unit.ALU, Unit.FMA) if self in (BitOp.SHR, BitOp.SHL) else (Unit.ALU,)

    @property
    def linear(self) -> bool:
        return self in (BitOp.SHR, BitOp.ROTR, BitOp.SHL, BitOp.LOP3)

    @property
    def issues(self) -> int:
        """Its instructions: a compare's or a select's two (an ISETP and a SEL), every other's one."""
        return 2 if self in (BitOp.CMP, BitOp.SEL) else 1


COMPARISONS = ("lt", "le", "gt", "ge", "eq", "ne")
"""A compare's predicates (`BitOp.CMP`), its `table`'s low three bits each's index (no other code is one);
plus 8 signed."""

_FILLED = {BitOp.LOGIC: 3, BitOp.MINMAX: 2, BitOp.CMP: 2}
"""The operands an instruction reads whatever values it is given, the immediate filling the rest."""


def operands(op: BitOp, values: Sequence, immediate) -> list:
    """The operands `op` reads given `values` and its `immediate`: LOP3 three, a min, max or compare two, the
    immediate each one past the values (PTX's `lop3.b32 d, a, imm, imm`); every other its values. The one
    declaration execution (`_run`), emission (`_written`) and the synthesizer's proof
    (tools/superoptimize_bits.py) read, whatever their values are (arrays, registers' names, bit-vectors)."""
    filled = _FILLED.get(op)
    return list(values) if filled is None else [*values, *[immediate] * (filled - len(values))]


class BitStep(NamedTuple):
    """One instruction of a form: `op` over the values at `args` (0 the stored word, `i + 1` step `i`'s), its
    immediate (a shift's amount, a mask, an operand), the unit it issues on, and `table` (LOGIC's truth table
    over its operands' tables 0xF0, 0xCC and 0xAA; IADD3's operands negated, a bit apiece)."""

    op: BitOp
    args: tuple[int, ...]
    immediate: int
    unit: Unit
    table: int = 0


def _run[Words: (np.ndarray, jax.Array)](steps: Sequence[BitStep], outputs: Sequence[int], words: Words,
                                          xp) -> tuple[Words, ...]:
    """Each output (`outputs`, 0 the stored word, `i + 1` step `i`'s) of `steps` over `words`, uint32 arrays
    of `xp` (NumPy, or `jax.numpy` for a kernel's portable path): the one evaluation of a form's steps, as
    each instruction computes it, its shifts in 32 bits."""
    # Each value the type of `words`, which `xp`'s operations keep and no checker follows through them.
    values: list = [words]
    for step in steps:
        first = values[step.args[0]] if step.args else xp.zeros_like(words)
        amount = xp.uint32(step.immediate)
        given = [values[arg] for arg in step.args]
        read = operands(step.op, given, xp.full_like(first, amount))
        if step.op is BitOp.SHR:
            value = first >> amount
        elif step.op is BitOp.ROTR:
            value = (first >> amount) | (first << xp.uint32(WORD - step.immediate))
        elif step.op is BitOp.SHL:
            value = first << amount
        elif step.op is BitOp.LOP3:
            value = (first if len(step.args) == 1 else first ^ values[step.args[1]]) & amount
        elif step.op is BitOp.LOGIC:
            value = _looked_up(step.table, read, xp)
        elif step.op is BitOp.IADD3:
            value = amount + sum((xp.uint32(0) - operand if step.table >> index & 1 else operand)
                                 for index, operand in enumerate(given))
        elif step.op is BitOp.IMAD:
            value = first * amount + (given[1] if len(given) > 1 else xp.uint32(0))
        elif step.op is BitOp.MINMAX:
            pair = read
            if step.table & 1:
                pair = [value.astype(xp.int32) for value in pair]
            value = (xp.maximum if step.table & 2 else xp.minimum)(*pair)
        elif step.op is BitOp.SAR:
            value = first.astype(xp.int32) >> xp.int32(step.immediate)
        elif step.op in (BitOp.SHRV, BitOp.SHLV):
            places = given[1]
            shifted = first >> xp.minimum(places, xp.uint32(WORD - 1)) if step.op is BitOp.SHRV \
                else first << xp.minimum(places, xp.uint32(WORD - 1))
            value = xp.where(places < xp.uint32(WORD), shifted, xp.uint32(0))
        elif step.op is BitOp.CMP:
            left, right = read
            if step.table & 8:
                left, right = left.astype(xp.int32), right.astype(xp.int32)
            compared = {"lt": left < right, "le": left <= right, "gt": left > right, "ge": left >= right,
                        "eq": left == right, "ne": left != right}[COMPARISONS[step.table & 7]]
            value = compared.astype(xp.uint32)
        elif step.op is BitOp.SEL:
            value = xp.where(first != xp.uint32(0), given[1], given[2])
        else:
            value = _permuted(given, step.immediate, xp)
        values.append(value.astype(xp.uint32))
    return tuple(values[index] for index in outputs)


def _looked_up[Words: (np.ndarray, jax.Array)](table: int, operands: Sequence[Words], xp) -> Words:
    """LOP3's `table` over its three `operands` (the first's truth table 0xF0, the second's 0xCC, the third's
    0xAA), as the sum of its minterms: each set bit of the table the AND of each operand or its complement."""
    a, b, c = operands
    value = xp.zeros_like(a)
    for minterm in range(8):
        if table >> minterm & 1:
            value |= ((a if minterm & 4 else ~a) & (b if minterm & 2 else ~b) & (c if minterm & 1 else ~c))
    return value


def _permuted[Words: (np.ndarray, jax.Array)](operands: Sequence[Words], selector: int, xp) -> Words:
    """PRMT's default mode over `operands` (the second the high word; the first again where it has one): each
    byte `k` of the result the byte that `selector`'s nibble `k` names of the eight (0-3 the first's)."""
    low, high = operands[0], operands[-1]
    value = xp.zeros_like(low)
    for k in range(4):
        source = selector >> 4 * k & 7
        word = low if source < 4 else high
        value |= (word >> xp.uint32(8 * (source % 4)) & xp.uint32(0xFF)) << xp.uint32(8 * k)
    return value


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
        return {unit: sum(step.op.issues for step in self.steps if step.unit is unit) for unit in Unit}

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
            lines.append(_written(step, names[value], [names[arg] for arg in step.args]))
        lines.append(f"mov.b32 $0, {names[self.outputs[output]]};")
        return "{\n" + "\n".join(lines) + "\n}"

    @property
    def price(self) -> tuple[int, int]:
        """Its busiest unit's issues, then its instructions (`Unit`: one rate): a decoder's words are many
        and independent, so its pipes' throughput prices it, not its chain's latency."""
        return max(self.issues.values()), sum(step.op.issues for step in self.steps)


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
"""Each linear instruction as the PTX written for its pipe (`_written` the others, which read a step's
table)."""


def _written(step: BitStep, out: str, ins: Sequence[str]) -> str:
    """Step `step` as PTX into `out` from `ins`: a linear one's `_PTX`; IADD3 as `add`/`sub` (ptxas issues one
    IADD3), IMAD as `mad.lo`, UMIN as `min.u32`, SHRV as `shr.b32` by a register (PTX clamps its amount at
    32), PRMT and LOGIC as themselves, the immediate their last operand."""
    imm = f"{step.immediate:#x}"
    if step.op.linear:
        return _PTX[step.op, step.unit](out, ins, step.immediate)
    if step.op is BitOp.LOGIC:
        return f"lop3.b32 {out}, {', '.join(operands(step.op, ins, imm))}, {step.table:#04x};"
    if step.op is BitOp.IADD3:
        lines = [f"mov.b32 {out}, {imm};"]
        lines += [f"{'sub' if step.table >> index & 1 else 'add'}.u32 {out}, {out}, {value};"
                  for index, value in enumerate(ins)]
        return " ".join(lines)
    if step.op is BitOp.IMAD:
        return f"mad.lo.u32 {out}, {ins[0]}, {imm}, {ins[1] if len(ins) > 1 else 0};"
    if step.op is BitOp.MINMAX:
        kind = f"{'max' if step.table & 2 else 'min'}.{'s32' if step.table & 1 else 'u32'}"
        return f"{kind} {out}, {', '.join(operands(step.op, ins, imm))};"
    if step.op is BitOp.SAR:
        return f"shr.s32 {out}, {ins[0]}, {step.immediate};"
    if step.op in (BitOp.SHRV, BitOp.SHLV):
        return f"{'shr' if step.op is BitOp.SHRV else 'shl'}.b32 {out}, {ins[0]}, {ins[1]};"
    if step.op is BitOp.CMP:
        kind = f"{COMPARISONS[step.table & 7]}.{'s32' if step.table & 8 else 'u32'}"
        left, right = operands(step.op, ins, imm)
        return (f"{{ .reg .pred p; setp.{kind} p, {left}, {right}; "
                f"selp.b32 {out}, 1, 0, p; }}")
    if step.op is BitOp.SEL:
        return f"{{ .reg .pred p; setp.ne.u32 p, {ins[0]}, 0; selp.b32 {out}, {ins[1]}, {ins[2]}, p; }}"
    return f"prmt.b32 {out}, {ins[0]}, {ins[-1]}, {imm};"
"""Each instruction as the PTX written for its pipe: a shift or rotation as `shr`/`shl`/`shf`, which ptxas
issues as SHF on the ALU, and a multiply by a power of two as `mul`, which it issues as IMAD on the FMA pipe
(PTX ISA, "Integer Arithmetic Instructions"; Nsight Compute's pipelines). PTX fixes the arithmetic, not the
SASS: ptxas chooses the final instructions, which the consumer's compile report shows."""


class Unproven(ValueError):
    """A form that does not compute its transform on some word."""


def proven(transform: BitTransform, form: BitForm) -> None:
    """Prove `form` computes `transform` on every input, or raise `Unproven`: a form of linear instructions
    (`BitOp.linear`) and its encoder are linear maps, and two linear maps that agree on each basis word agree
    on all 2^width; the input's every word is checked besides where it has 16 bits or fewer. Bits outside a
    register's mask are proven zero by the same argument. A fixed layout's encoder must be the identity: its
    form reads the input as stored. A form with an instruction that is not linear is proven against its
    reference on every word instead (its consumer's test), never here."""
    if not all(step.op.linear for step in form.steps):
        raise Unproven(f"{transform.name}: a form not linear proves on every word, not on a basis")
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


class Minifloat(NamedTuple):
    """A narrow float an encode rounds fp32 to (`encoded`): its exponent and mantissa bits, its bias, its
    largest finite code (the saturated one) and the code a NaN takes (None where it has none: e2m1, whose
    NaN saturates as the largest's)."""

    exponent: int
    mantissa: int
    bias: int
    largest: int
    nan: int | None

    @property
    def width(self) -> int:
        return 1 + self.exponent + self.mantissa

    @property
    def largest_float32(self) -> int:
        """The largest finite code's value as fp32 bits: what a value past it saturates to."""
        exponent, fraction = self.largest >> self.mantissa, self.largest & ((1 << self.mantissa) - 1)
        return (exponent - self.bias + 127) << 23 | fraction << (23 - self.mantissa)


E4M3 = Minifloat(exponent=4, mantissa=3, bias=7, largest=0x7E, nan=0x7F)
"""FP8 e4m3fn: no infinity, its largest finite 448 (0x7E), NaN 0x7F."""

E5M2 = Minifloat(exponent=5, mantissa=2, bias=15, largest=0x7B, nan=0x7F)
"""FP8 e5m2: its largest finite 57344 (0x7B), NaN 0x7F as the saturating cvt keeps it."""

E2M1 = Minifloat(exponent=2, mantissa=1, bias=1, largest=0x7, nan=None)
"""FP4 e2m1 (NVFP4's code): its largest 6 (0x7), no NaN."""


def encoded[Words: (np.ndarray, jax.Array)](x: Words, minifloat: Minifloat, xp=jnp) -> Words:
    """fp32 `x` (as uint32 bits) rounded to `minifloat`'s codes, uint32 a code: to nearest, ties to even,
    saturating finite (an infinity, or a value past the largest, as the largest), a NaN as its NaN code
    (the largest's where it has none), as `cvt.rn.satfinite` gives it where a family has the instruction.
    Integer arithmetic alone (a family without the cvt, sm80's and sm86's, issues it so): the magnitude
    clipped to the largest as an unsigned minimum, then one rounding shift of the fp32 exponent and fraction
    for a normal result, of the significand with its implicit bit past the subnormal's place otherwise, its
    carry into the exponent the rounding's own."""
    word = xp.uint32
    sign = x >> word(31)
    magnitude = xp.minimum(x & word(0x7FFFFFFF), word(minifloat.largest_float32))
    nan = (x & word(0x7FFFFFFF)) > word(0x7F800000)
    biased = (magnitude >> word(23)).astype(xp.int32)
    exponent = biased - (127 - minifloat.bias)
    fraction = magnitude & word(0x7FFFFF)
    normal = exponent >= 1
    significand = xp.where(normal, word(0), (biased != 0).astype(word) << word(23)) | fraction
    value = xp.where(normal, exponent.astype(word) << word(23) | fraction, significand)
    beyond = xp.minimum(23 - minifloat.mantissa + 1 - exponent, 31)
    shift = xp.where(normal, 23 - minifloat.mantissa, beyond).astype(word)
    half = (word(1) << (shift - word(1))) - word(1)
    code = (value + half + (value >> shift & word(1))) >> shift
    nan_code = minifloat.largest if minifloat.nan is None else minifloat.nan
    code = xp.where(nan, word(nan_code), code)
    return code | sign << word(minifloat.width - 1)



def encode_form(minifloat: Minifloat) -> BitForm:
    """`encoded`'s instruction form for `minifloat`, a word to its code: the reference traced into the
    vocabulary and superoptimized (`tools/superoptimize_bits.py`: each window of up to three steps replaced
    by a cheaper program of one or two that Z3 proves equal on every input), from 41 issues to 37 for each
    of e4m3, e5m2 and e2m1, the same program but for the minifloat's constants (its largest finite value,
    its bias, its mantissa's width, its NaN code and its sign's place). Proven against `encoded` on every
    fp32 word by its test; its PTX (`BitForm.ptx`) a kernel's inline encode on a family with no satfinite
    cvt (sm80, sm86), where its consumer's timing decides it against the compiler's lowering of `encoded`."""
    m = minifloat.mantissa
    nan = minifloat.largest if minifloat.nan is None else minifloat.nan
    alu = Unit.ALU
    steps = (
        BitStep(BitOp.SHR, (0,), 31, alu),                                  # 1 the sign
        BitStep(BitOp.LOGIC, (0,), 0x7FFFFFFF, alu, 0xC0),                  # 2 the magnitude
        BitStep(BitOp.MINMAX, (2,), minifloat.largest_float32, alu),        # 3 saturated finite
        BitStep(BitOp.CMP, (2,), 0x7F800000, alu, 2),                       # 4 a NaN
        BitStep(BitOp.SHR, (3,), 23, alu),                                  # 5 its biased exponent
        BitStep(BitOp.IADD3, (5,), -(127 - minifloat.bias) & MASK, alu),    # 6 rebiased
        BitStep(BitOp.LOGIC, (3,), 0x7FFFFF, alu, 0xC0),                    # 7 its fraction
        BitStep(BitOp.CMP, (6,), 1, alu, 11),                               # 8 a normal result
        BitStep(BitOp.MINMAX, (5,), 1, alu),                                # 9 a normal input
        BitStep(BitOp.SHL, (9,), 23, alu),                                  # 10 its implicit bit
        BitStep(BitOp.LOGIC, (10, 7), 0, alu, 0xFC),                        # 11 its significand
        BitStep(BitOp.SEL, (8, 7, 11), 0, alu),
        BitStep(BitOp.SHL, (6,), 23, alu),
        BitStep(BitOp.LOGIC, (13, 7), 0, alu, 0xFC),
        BitStep(BitOp.SEL, (8, 14, 12), 0, alu),                            # 15 the value rounded
        BitStep(BitOp.IADD3, (6,), 24 - m, alu, 1),
        BitStep(BitOp.MINMAX, (16,), 31, alu, 1),
        BitStep(BitOp.IADD3, (), 23 - m, alu),
        BitStep(BitOp.SEL, (8, 18, 17), 0, alu),                            # 19 its shift
        BitStep(BitOp.IADD3, (19,), MASK, alu),
        BitStep(BitOp.IADD3, (), 1, alu),
        BitStep(BitOp.SHLV, (21, 20), 0, alu),
        BitStep(BitOp.IADD3, (22, 15), MASK, alu),                          # 23 plus half less one
        BitStep(BitOp.SHRV, (15, 19), 0, alu),
        BitStep(BitOp.LOGIC, (24,), 1, alu, 0xC0),                          # 25 the tie's bit
        BitStep(BitOp.IADD3, (23, 25), 0, alu),
        BitStep(BitOp.SHRV, (26, 19), 0, alu),                              # 27 the code
        BitStep(BitOp.IADD3, (), nan, alu),
        BitStep(BitOp.SEL, (4, 28, 27), 0, alu),
        BitStep(BitOp.SHL, (1,), minifloat.width - 1, alu),
        BitStep(BitOp.LOGIC, (29, 30), 0, alu, 0xFC),                       # 31 signed
    )
    return BitForm(steps, (len(steps),), tuple(1 << bit for bit in range(WORD)))

def packed[Words: (np.ndarray, jax.Array)](codes: Words, minifloat: Minifloat, xp=jnp) -> Words:
    """`codes` (uint32 a code, `[..., n]`) packed into uint32 words along the last axis, `[..., n * width /
    32]`: e4m3's and e5m2's four a word little-endian (code 0 in the low byte, as `cvt ...x2` pairs and a
    bf16-ordered store give them), e2m1's eight as NVFP4's stored word (`NVFP4_BFLOAT16_FORM`'s encoder of
    the word holding code `i` at bits `4 i`), the word `nvfp4_decoded` reads."""
    per = 32 // minifloat.width
    grouped = codes.reshape(*codes.shape[:-1], -1, per)
    places = xp.arange(per, dtype=xp.uint32) * xp.uint32(minifloat.width)
    # The codes' bits are disjoint, so their sum is their union.
    natural = xp.sum(grouped << places, axis=-1, dtype=xp.uint32)
    return _parities(natural, NVFP4_BFLOAT16_FORM.encoder, xp) if minifloat is E2M1 else natural


def nvfp4_quantized(x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """An fp32 tile `x` `[depth, columns]` as NVFP4: its packed codes (`[depth / 8, columns]` uint32, the
    stored order `nvfp4_decoded` reads), its block scales (`[depth / 16, columns]` float8_e4m3fn, one for
    each sixteen along the depth) and its tensor scale (fp32): the tensor scale its largest magnitude over
    6 * 448 (1 where it is zero), so every block's scale fits e4m3; a block's scale the least e4m3 at or
    above its largest magnitude over 6 times the tensor scale, so no code saturates; each value over its
    block's scale and the tensor scale rounded to e2m1 (`encoded`), a block of zeros codes of zero. Its
    decode (`nvfp4_decoded`) times `NVFP4_OWED` times the tensor scale is each value's rounding, as the GEMM
    epilogue's NVFP4 output stores it (row 4050)."""
    magnitude = jnp.abs(x)
    largest = jnp.max(magnitude)
    tensor = jnp.where(largest > 0, largest / np.float32(6 * 448), np.float32(1))
    blocks = jnp.max(magnitude.reshape(-1, 16, x.shape[-1]), axis=1)
    ratio = blocks / (np.float32(6) * tensor)
    nearest = encoded(jax.lax.bitcast_convert_type(ratio, jnp.uint32), E4M3)

    def value(codes: jax.Array) -> jax.Array:
        return jax.lax.bitcast_convert_type(codes.astype(jnp.uint8), jnp.float8_e4m3fn).astype(jnp.float32)

    # The nearest rounded up where it fell below the ratio, never past the largest finite (448).
    codes = jnp.where(value(nearest) < ratio, jnp.minimum(nearest + jnp.uint32(1), jnp.uint32(E4M3.largest)),
                      nearest)
    scales = jax.lax.bitcast_convert_type(codes.astype(jnp.uint8), jnp.float8_e4m3fn)
    each = jnp.repeat(value(codes), 16, axis=0) * tensor
    quotient = jnp.where(each > 0, x / jnp.where(each > 0, each, 1), 0)
    words = packed(encoded(jax.lax.bitcast_convert_type(quotient, jnp.uint32), E2M1).T, E2M1).T
    return words, scales, tensor
