"""Dew: what kind of call a grouped product is (`Kind`: what it makes, `Made`), which with the device's
instruction family is its cell in `grouped_matmul`'s selector (`grouped_matmul.DECLARED`), and the receipts
that timed the dense products holding cells of their own (`MEASURED`), each as Dew's records name it."""

import enum
from typing import NamedTuple


class Made(enum.Enum):
    """What a grouped call makes (`Kind`): a forward product, its input gradient (the matrix read transposed),
    or the matrices' kernel gradient, summed by the tensor cores or exactly (`exact_sums`)."""

    FORWARD = 'forward'
    INPUT_GRADIENT = 'input_gradient'
    KERNEL_GRADIENT = 'kernel_gradient'
    EXACT_KERNEL_GRADIENT = 'kernel_gradient exact'


class Kind(NamedTuple):
    """The kind of call a grouped product is (`grouped_matmul.GroupedCase.kind`), which with the device's
    instruction family is its cell (`grouped_matmul.DECLARED`): dense (one matrix, every row) or grouped, what
    it makes, the operands no receipt of a 16-bit call times: FP8 ones (XLA's dot of them sums narrower than
    the FP8 contract holds, row 1625), rows gathered through a table or stored through one (no old route reads
    them); and, for a dense product a receipt timed on its family (`MEASURED`: its matrix, over the rows it
    timed), that matrix's `[in, out]` extents, a cell of its own, which the receipt's winner for it holds
    (None for every other product, the kind's cell)."""

    dense: bool
    made: Made
    fp8: bool = False
    gathered: bool = False
    scattered: bool = False
    matrix: tuple | None = None
    formed: bool = False
    """Whether its left operand is formed by a prologue the kernel runs tile by tile (a norm's normalization
    the algebra's recognition fused into the product, `fused_gemm`), its call one that reads the rows the
    prologue forms them from: a forward's or a kernel gradient's, a cell of its own where a receipt timed one
    (`MEASURED`)."""
    together: bool = False
    """Whether it is several products in one launch: matrices side by side (`gmm_call`'s `beside`, a layer's
    query, key and value), or their input gradient's products chained over their joined depths (`chained`),
    a cell of its own, which no one product's receipt times (head review of 4ddd0d7ea F1: the ordinary
    products' cells stood for them); its `matrix` their geometry where a receipt timed them (`TOGETHER`: the
    launch's form, the shared extent and each product's width in the launch's order)."""

    def __str__(self) -> str:
        parts = (('fp8',) * self.fp8 + ('gathered',) * self.gathered + ('scattered',) * self.scattered
                 + ('formed',) * self.formed + ('together',) * self.together)
        timed = () if self.matrix is None else ('x'.join(map(str, self.matrix)),)
        return ' '.join(('dense' if self.dense else 'grouped', self.made.value, *parts, *timed))


FOUR_B = ((2560, 9216), (9216, 2560), (2560, 1024), (2560, 64))
"""Qwen3.5-4B's dense projections' matrices `[in, out]`: its MLP's up and down, its attention's key, its
gated delta net's gates, which M-EM-36-gh200 and em35d timed at 24,576 tokens, each product of each."""

class Timed(NamedTuple):
    """The dense products a kind's receipts timed (`MEASURED`): of these `matrices` `[in, out]`, over `rows`
    from `fewest` to `most` tokens. A receipt is evidence for the products it measured alone: a product of
    another matrix, or of one of these over rows outside them, takes its kind's cell (row 6794: the 4B
    key's cells, timed at 24,576 tokens, took a fine-tune's key and value products at 2,048, whose 216
    template launches a step ran 6.57 ms, the step's dense products 1.75 ms slower than cuBLAS's alone)."""

    matrices: frozenset[tuple]
    fewest: int
    most: int

    def holds(self, matrix: tuple, rows: int) -> bool:
        """Whether a product of `matrix` over `rows` tokens is one these receipts timed."""
        return matrix in self.matrices and self.fewest <= rows <= self.most


NORMED_08B = ((1024, 4096 + 512 + 512), (1024, 2 * 3584))
"""Qwen3.5-0.8B's normed projections M-EM-65 timed (`M_EM_65`): the attention input norm before the joined
query (doubled by its output gate), key and value, and the MLP's norm before gate and up, `[in, out]`."""

NORMED_4B_8B = ((2560, 8192 + 1024 + 1024), (2560, 2 * 9216), (4096, 4096 + 1024 + 1024), (4096, 2 * 14336))
"""Qwen3.5-4B's and Llama-3.1-8B's normed projections M-EM-67 timed (`M_EM_67`): each's joined query (the
4B's doubled by its output gate), key and value, and its gate and up, `[in, out]`."""

M_EM_67 = ("M-EM-67-gh200 (a GH200, main 27bc19ccba3, rounds.py's two arms interleaved 5 rounds, fresh "
           "caches an arm, 2 x 2048 and 4 x 3072 rows of bf16, a training step's forward and backward): the "
           "norm in the template's prologue against the fused norm kernel then XLA's dot, 0.562x to 0.624x "
           "at every projection, ranges apart; an A100 1.041x to 1.100x, so no A100 cell")
M_EM_65 = ("M-EM-65-a100 and M-EM-65-gh200 (an A100 and a GH200, main 99c7334e2, rounds.py's two arms "
           "interleaved 5 rounds, CUPTI medians, 4 x 3072 rows of bf16, a training step's forward and "
           "backward): the norm in the template's prologue against the fused norm kernel then XLA's dot, "
           "0.955x and 0.619x at the attention norm + q/k/v, 0.971x and 0.533x at the MLP norm + gate/up, "
           "ranges apart")
MEASURED: dict[tuple[str | None, Made, bool], tuple[Timed, ...]] = {
    **{(family, made, False): (Timed(frozenset(FOUR_B), 24576, 49152),) for family in ('wgmma', 'tcgen05')
       for made in (Made.FORWARD, Made.INPUT_GRADIENT, Made.KERNEL_GRADIENT)},
    **{('mma.sync', made, True): (Timed(frozenset(NORMED_08B), 12288, 12288),)
       for made in (Made.FORWARD, Made.KERNEL_GRADIENT)},
    **{('wgmma', made, True): (Timed(frozenset(NORMED_08B), 12288, 12288),
                               Timed(frozenset(NORMED_4B_8B), 4096, 12288))
       for made in (Made.FORWARD, Made.KERNEL_GRADIENT)}}
"""The dense products each instruction family's receipts timed, by what the product makes: one of them is a
cell of its own (`Kind.matrix`), which its receipt's winner holds, each receipt its own matrices and rows;
every other product its kind's. The
receipts timed 24,576 tokens (M-EM-36, M-EM-52, em35d); the 4B step's 49,152 (four rows) held the same
cells at parity with cuBLASLt (M-EM-54, interleaved in one session: 3,418.7 against 3,416.9 ms)."""

QKV_8B_4B = (('chained', 4096, (1024, 1024, 4096)), ('chained', 2560, (1024, 1024, 8192)))
"""Llama-3.1-8B's and Qwen3.5-4B's query, key and value input gradient as M-EM-73 timed it chained: the
launch's form, the features, and its products' widths in the launch's order (the value's, the key's, the
query's: the reverse pass's, `gemm._one_launch`; the 4B's query doubled by its output gate)."""

M_EM_73 = ("M-EM-73-gh200 (a GH200, base/kernels b1ab251ac, CUPTI medians of 5 rounds): the query, key "
           "and value's bf16 input gradient chained in one launch 0.80x XLA's (cuBLAS's) three products at "
           "Llama-3.1-8B's 4 x 3072 rows and 0.87x at Qwen3.5-4B's, 0.837x the template's three products "
           "each added in the next's epilogue at the 8B's (0.815 against 0.974 ms); 1.32x XLA's at the 4B's "
           "1 x 2048, provisional (n=5, spreads to 26%)")

class Together(NamedTuple):
    """A receipt of products in one launch (`Kind.together`) and the kernel it timed winning: of what family
    and what they make (`made`), their operands' `dtype`, the geometries and rows it timed (`timed`), the
    kernel (`winner`, a grouped selector's kernel by name) and the receipt (`evidence`): the one record a
    call's cell (`grouped_matmul.grouped_case`) and its declared route (`grouped_matmul.DECLARED`) both read
    (base review 760be77fe F6)."""

    family: str
    made: Made
    dtype: str
    timed: Timed
    winner: str
    evidence: str


QKV_4B_SHORT = (('chained', 2560, (1024, 1024, 8192)),)
"""Qwen3.5-4B's query, key and value chained input gradient, where M-EM-73 measured the template behind XLA at
1 x 2048 (0.263 vs 0.200 ms, 1.32x; rows 9665, 9768, 10151)."""

TOGETHER = (
    Together('wgmma', Made.INPUT_GRADIENT, 'bfloat16',
             Timed(frozenset(QKV_4B_SHORT), 2048, 2048), 'apart', M_EM_73),
)
"""The products in one launch receipts timed losing to running apart (`Together`): declared to `'apart'`
(`grouped_matmul.DECLARED`); every other products-in-one-launch call defaults to the template
(kernel-templates.md §7; rows 9665, 9768)."""


M_EM_36 = ("M-EM-36-gh200 (a GH200, base/kernels df82480cb5c, CUPTI medians of 5 rounds, cuBLASLt autotuned "
           "at run, 24,576 tokens)")
M_EM_52 = ("M-EM-52-gh200 (a GH200 under its power cap, base/kernels 3bea6245b56's derivation: the kernel "
           "gradient on a multicast pair, CUPTI medians of 5 rounds interleaved with cuBLASLt's, autotuned "
           "in the same process, 24,576 tokens)")
EM35D = ("em35d (a B200, base/kernels ce6eff7f0be, CUPTI medians of 5 rounds, cuBLASLt autotuned at run, "
         "24,576 tokens)")
B200_GRID = ("the B200 grid (base/kernels 5ac163abbec, medians of 5 rounds' same-round "
             "ratios, cuBLASLt autotuned at run, 24,576 tokens; its down kernel gradient the template's, "
             "0.833x)")
