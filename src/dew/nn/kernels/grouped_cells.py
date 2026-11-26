"""Dew: what kind of call a grouped product is (`Kind`: what it makes, `Made`), which with the device's
generation is its cell in `grouped_matmul`'s selector (`grouped_matmul.DECLARED`), and the receipts that
timed the products holding cells (`MEASURED`, `SCOREBOARDS`), each as Dew's records name it. A cell is a
generation's, not an instruction family's: one family's generations part in speed against their references
(mma.sync's A100 forward at 1.000x XLA's dot where an A10's is 1.28x), so a receipt is evidence for the
generation it ran on alone, and a generation no receipt timed takes the template (bible §6, 2026-11-24:
"templates should replace ref routes on parity")."""

import enum
import math
from collections.abc import Mapping
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
    generation is its cell (`grouped_matmul.DECLARED`): dense (one matrix, every row) or grouped, what it
    makes, the operands no receipt of a 16-bit call times: FP8 ones (XLA's dot of them sums narrower than the
    FP8 contract holds, row 1625), rows gathered through a table or stored through one (no old route reads
    them); and, for a dense product a receipt timed on its generation (`MEASURED`: its matrix, over the rows
    it timed), that matrix's `[in, out]` extents and the rows the receipt timed (`span`), a cell of its own,
    which the receipt's winner for it holds (None for every other product, the kind's cell)."""

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
    span: tuple[int, int] | None = None
    """The fewest and most rows the receipt timing its `matrix` timed (`Timed.span`): one matrix's products
    over another receipt's rows are a cell of their own (the 4B's up input gradient: the template's at 24,576
    tokens, M-EM-36, XLA's at 8,192, M-NG-45b), None where no receipt timed it."""

    def __str__(self) -> str:
        parts = (('fp8',) * self.fp8 + ('gathered',) * self.gathered + ('scattered',) * self.scattered
                 + ('formed',) * self.formed + ('together',) * self.together)
        rows = () if self.span is None else (f'at {self.span[0]}-{self.span[1]} rows',)
        timed = () if self.matrix is None else ('x'.join(map(str, self.matrix)), *rows)
        return ' '.join(('dense' if self.dense else 'grouped', self.made.value, *parts, *timed))


FOUR_B = ((2560, 9216), (9216, 2560), (2560, 1024), (2560, 64))
"""Qwen3.5-4B's dense projections' matrices `[in, out]`: its MLP's up and down, its attention's key, its
gated delta net's gates, which M-EM-36-gh200 and em35d timed at 24,576 tokens, each product of each (the 4B
step's 49,152, four rows, held the same cells at parity with cuBLASLt, M-EM-54: 3,418.7 against 3,416.9
ms)."""

class Timed(NamedTuple):
    """The dense products a kind's receipts timed (`MEASURED`): of these `matrices` `[in, out]`, over `rows`
    from `fewest` to `most` tokens, of operands of `dtype` (the arithmetic it timed: a float32 call's TF32
    kernel is no bf16 receipt's). A receipt is evidence for the products it measured alone: a product of
    another matrix, or of one of these over rows outside them, takes its kind's cell (row 6794: the 4B
    key's cells, timed at 24,576 tokens, took a fine-tune's key and value products at 2,048, whose 216
    template launches a step ran 6.57 ms, the step's dense products 1.75 ms slower than cuBLAS's alone)."""

    matrices: frozenset[tuple]
    fewest: int
    most: int
    dtype: str = 'bfloat16'
    stored: str | None = None
    """The output's dtype it timed, which a launch's staging and memory read; None where its record states
    none."""

    def holds(self, matrix: tuple, rows: int, dtype: str = 'bfloat16', stored: str | None = None) -> bool:
        """Whether a product of `matrix` over `rows` tokens of `dtype` operands, stored in `stored`, is one
        these receipts timed."""
        return (matrix in self.matrices and self.fewest <= rows <= self.most and dtype == self.dtype
                and self.stored in (None, stored))

    @property
    def span(self) -> tuple[int, int]:
        """The rows these receipts timed, a cell's (`Kind.span`)."""
        return self.fewest, self.most


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
PROJECTIONS = {
    'llama-3-8b qkv': (4096, 6144), 'llama-3-8b o': (4096, 4096), 'llama-3-8b gate_up': (4096, 28672),
    'llama-3-8b down': (14336, 4096), 'qwen3-8b gate_up': (4096, 24576), 'qwen3-8b down': (12288, 4096),
    'qwen3-32b qkv': (5120, 10240), 'qwen3-32b gate_up': (5120, 51200), 'qwen3-32b down': (25600, 5120),
    'llama-3-70b down': (28672, 8192), 'qwen3.5-4b up': (2560, 9216), 'qwen3.5-4b down': (9216, 2560),
    'qwen3.5-4b key': (2560, 1024)}
"""The scoreboard's dense projections (`SCOREBOARDS`), each its matrix `[in, out]`: the 13 projections the
GEMM scoreboard times at 8,192 tokens."""

Ranged = tuple[float, float, float]
"""A route's times over a receipt's rounds, in milliseconds: the least, the median and the most."""

STORED = {Made.FORWARD: 'bfloat16', Made.INPUT_GRADIENT: 'bfloat16', Made.KERNEL_GRADIENT: 'float32'}
"""The output dtype the dense receipts' products store (`SCOREBOARDS`, `FOUR_B_CELLS`): bf16 forwards and
input gradients from bf16 operands, fp32 kernel gradients (the scoreboard's harness, em70r gemm_cells; the
4B's dW into fp32, records/scoreboard.md)."""

TEMPLATE, REFERENCE = 'mosaic', 'xla'
"""The routes a scoreboard's cell goes to (`Scoreboard.winner`), by the grouped selector's kernels' names."""


class Timing(NamedTuple):
    """One product a receipt timed, the one record its cell is recognized by (`MEASURED`, `Kind.span`) and
    declared from (`grouped_matmul.DECLARED`): its `generation`, what it `made`, its `matrix` `[in, out]` and
    whether a prologue formed its rows (`formed`), the rows and arithmetic it was timed at (`timed`), the
    route that won it (`winner`) and the receipt (`evidence`)."""

    generation: str
    made: Made
    matrix: tuple[int, int]
    timed: Timed
    winner: str
    evidence: str
    formed: bool = False


class Scoreboard(NamedTuple):
    """One receipt's dense products on one GPU `generation` at `rows` tokens, the template against XLA's dot
    (cuBLAS/cuBLASLt, compiled and autotuned on the GPU), each product's routes' rounds and their compiled
    programs' temporaries (`times`: by what it made, each projection's template's then XLA's `Ranged`, then
    the two temporaries' bytes), and the receipt (`evidence`). A product is at parity where the template's
    rounds lie at or under XLA's, overlapping them or wholly under, holding no more temporaries
    (`at_parity`: kernel-templates.md §1, compute time and peak memory); a kind takes the template whole only
    once every product it timed is (bible §6, 2026-11-24: "templates should replace ref routes on parity"),
    and keeps XLA's dot otherwise, its lagging products its derivation's to climb (`winner`), its products at
    parity the template's, each a cell of its own over the receipt's rows (`timings`)."""

    generation: str
    rows: int
    evidence: str
    times: Mapping[Made, Mapping[str, tuple[Ranged, Ranged, tuple[int, int]]]]

    def geomean(self, made: Made) -> float:
        """The geometric mean of the template's median over XLA's at each product making `made`."""
        ratios = [template[1] / reference[1] for template, reference, _ in self.times[made].values()]
        return math.exp(sum(map(math.log, ratios)) / len(ratios))

    def at_parity(self, made: Made, projection: str) -> bool:
        """Whether the template's rounds at `projection` making `made` lie at or under XLA's (none of them
        past every one of XLA's) and its program's temporaries no more than XLA's."""
        template, reference, (held, theirs) = self.times[made][projection]
        return template[0] <= reference[2] and held <= theirs

    def winner(self, made: Made) -> str:
        """The route a kind making `made` takes: `TEMPLATE` where every product it timed is at parity
        (`at_parity`), else `REFERENCE`."""
        return TEMPLATE if all(self.at_parity(made, name) for name in self.times[made]) else REFERENCE

    def why(self, made: Made, projection: str | None = None) -> str:
        """The receipt's reading for `projection`'s cell, or for its kind's where None."""
        if projection is None:
            behind = sum(not self.at_parity(made, name) for name in self.times[made])
            return (f"{self.evidence}: the template behind XLA's dot at {behind} of {len(self.times[made])} "
                    f"projections at {self.rows} tokens, in time or temporaries (geomean "
                    f"{self.geomean(made):.3f}x)")
        template, reference, (held, theirs) = self.times[made][projection]
        apart = 'every round faster' if template[2] < reference[0] else 'rounds overlapping'
        return (f"{self.evidence}: {projection} the template {template[1] / reference[1]:.3f}x XLA's dot "
                f"({template[1]:.4f} against {reference[1]:.4f} ms, {apart}; temporaries {held} against "
                f"{theirs} bytes)")

    def timings(self) -> tuple[Timing, ...]:
        """Each product it timed (`Timing`), the template's where at parity, XLA's otherwise."""
        return tuple(Timing(self.generation, made, PROJECTIONS[name],
                            Timed(frozenset({PROJECTIONS[name]}), self.rows, self.rows, stored=STORED[made]),
                            TEMPLATE if self.at_parity(made, name) else REFERENCE, self.why(made, name))
                     for made, projections in self.times.items() for name in projections)


M_NG_45_A100 = Scoreboard('sm80', 8192, (
    "M-NG-45-a100 (A100-SXM4-40GB at 1095 MHz, train 1204 b71af0d11, rounds_window 5 rounds, CUPTI, XLA "
    "compiled and autotuned on the GPU)"), {
    Made.FORWARD: {
        'qwen3.5-4b key': ((0.2224, 0.2227, 0.2229), (0.2147, 0.2147, 0.2149), (0, 0)),
        'llama-3-8b o': ((1.2649, 1.2651, 1.2658), (1.2252, 1.2253, 1.2254), (0, 0)),
        'llama-3-8b qkv': ((1.9145, 1.9147, 1.9154), (1.9482, 1.9818, 1.9879), (0, 0)),
        'qwen3.5-4b up': ((1.8148, 1.8154, 1.8163), (1.8159, 1.8175, 1.8191), (0, 0)),
        'qwen3.5-4b down': ((1.7753, 1.7755, 1.7759), (1.6908, 1.6911, 1.6913), (0, 0)),
        'qwen3-32b qkv': ((3.9536, 3.9547, 3.9553), (3.8332, 3.8374, 3.8406), (0, 0)),
        'qwen3-8b down': ((3.7208, 3.7217, 3.7225), (3.6114, 3.6156, 3.6168), (0, 0)),
        'llama-3-8b down': ((4.3310, 4.3314, 4.3333), (4.2492, 4.2813, 4.2859), (0, 0)),
        'qwen3-8b gate_up': ((7.4923, 7.4951, 7.4963), (8.2202, 8.2209, 8.2237), (0, 805306624)),
        'qwen3-32b down': ((9.6919, 9.6925, 9.6940), (9.2246, 9.2283, 9.2345), (0, 0)),
        'llama-3-8b gate_up': ((8.7250, 8.7284, 8.7296), (9.5830, 9.5874, 9.5886), (0, 939524352)),
        'llama-3-70b down': ((17.1798, 17.1941, 17.2059), (16.3624, 16.3626, 16.3764), (0, 0)),
        'qwen3-32b gate_up': ((19.4758, 19.4830, 19.4869), (20.7995, 20.8091, 20.8232), (0, 1677721856)),
    },
    Made.INPUT_GRADIENT: {
        'qwen3.5-4b key': ((0.2340, 0.2344, 0.2349), (0.2239, 0.2243, 0.2245), (0, 0)),
        'llama-3-8b o': ((1.3573, 1.3580, 1.3586), (1.2392, 1.2398, 1.2400), (0, 0)),
        'llama-3-8b qkv': ((2.0158, 2.0167, 2.0170), (1.8564, 1.8593, 1.8665), (0, 0)),
        'qwen3.5-4b up': ((1.8075, 1.8077, 1.8104), (1.7154, 1.7156, 1.7219), (0, 0)),
        'qwen3.5-4b down': ((1.9071, 1.9080, 1.9088), (1.8739, 1.8747, 1.8753), (0, 0)),
        'qwen3-32b qkv': ((4.0035, 4.0097, 4.0104), (3.8142, 3.8252, 3.8264), (0, 0)),
        'qwen3-8b down': ((3.9205, 3.9212, 3.9216), (3.8102, 3.8180, 3.8207), (0, 0)),
        'llama-3-8b down': ((4.5875, 4.5890, 4.5896), (4.4804, 4.4823, 4.4863), (0, 0)),
        'qwen3-8b gate_up': ((7.9127, 7.9198, 7.9216), (7.3331, 7.3425, 7.3478), (0, 134217984)),
        'qwen3-32b down': ((10.1041, 10.1043, 10.1090), (10.5273, 10.5300, 10.5359), (0, 838861056)),
        'llama-3-8b gate_up': ((9.2100, 9.2122, 9.2124), (8.7936, 8.7952, 8.7967), (0, 134217984)),
        'llama-3-70b down': ((17.8228, 17.8280, 17.8310), (17.8540, 17.8571, 17.8617), (0, 939524352)),
        'qwen3-32b gate_up': ((29.9618, 30.2808, 35.7339), (19.0651, 19.0673, 19.0764), (0, 167772416)),
    },
    Made.KERNEL_GRADIENT: {
        'qwen3.5-4b key': ((0.3050, 0.3053, 0.3053), (0.2909, 0.2911, 0.2917), (0, 46137600)),
        'llama-3-8b o': ((1.4180, 1.4183, 1.4186), (1.3684, 1.3693, 1.3699), (0, 71303424)),
        'llama-3-8b qkv': ((2.1240, 2.1243, 2.1252), (1.9759, 1.9765, 1.9771), (0, 71303424)),
        'qwen3.5-4b up': ((2.0046, 2.0050, 2.0051), (1.8309, 1.8322, 1.8326), (0, 46137600)),
        'qwen3.5-4b down': ((1.9968, 1.9973, 1.9980), (1.9903, 1.9907, 1.9914), (0, 155189504)),
        'qwen3-32b qkv': ((4.2638, 4.2648, 4.2650), (3.8879, 3.8889, 3.8894), (0, 88080640)),
        'qwen3-8b down': ((4.0862, 4.0867, 4.0881), (3.9270, 3.9270, 3.9285), (0, 205521152)),
        'llama-3-8b down': ((4.7754, 4.7768, 4.7796), (4.5889, 4.5892, 4.5904), (0, 239075584)),
        'qwen3-8b gate_up': ((8.0055, 8.0066, 8.0076), (7.2275, 7.2311, 7.2317), (0, 71303424)),
        'llama-3-8b gate_up': ((9.3850, 9.3855, 9.3877), (8.4450, 8.4454, 8.4459), (0, 71303424)),
        'qwen3-32b down': ((10.5439, 10.5595, 10.5661), (9.9708, 9.9752, 9.9781), (0, 423624960)),
        'llama-3-70b down': ((18.6177, 18.6187, 18.6195), (17.2770, 17.2828, 17.2841), (0, 473956608)),
        'qwen3-32b gate_up': ((20.9652, 20.9672, 20.9758), (18.6990, 18.7110, 18.7178), (0, 88080640)),
    },
})
M_NG_27_GH200 = Scoreboard('sm90', 8192, (
    "M-NG-27-gh200 (GH200, lane/gemm 50232e845c7: no balanced tail on a multi-wave cluster launch, "
    "rounds_window 5 rounds, CUPTI, XLA compiled and autotuned on the GPU)"), {
    Made.FORWARD: {
        'qwen3.5-4b key': ((0.0541, 0.0717, 0.0733), (0.0584, 0.0731, 0.0752), (0, 0)),
        'llama-3-8b o': ((0.3374, 0.3790, 0.3993), (0.3599, 0.4082, 0.4525), (0, 0)),
        'llama-3-8b qkv': ((0.4984, 0.5311, 0.5410), (0.5468, 0.5656, 0.5682), (0, 0)),
        'qwen3.5-4b up': ((0.4744, 0.5000, 0.5092), (0.5404, 0.5531, 0.5587), (0, 0)),
        'qwen3.5-4b down': ((0.4779, 0.4819, 0.4850), (0.4978, 0.5071, 0.5183), (0, 0)),
        'qwen3-32b qkv': ((1.0420, 1.0602, 1.0645), (1.1764, 1.1807, 1.1931), (0, 0)),
        'qwen3-8b down': ((1.0919, 1.0960, 1.1263), (1.1808, 1.1898, 1.1963), (0, 0)),
        'llama-3-8b down': ((1.2908, 1.3012, 1.3082), (1.3574, 1.3766, 1.3858), (0, 0)),
        'qwen3-8b gate_up': ((2.1658, 2.2094, 2.2440), (2.5900, 2.6274, 2.6574), (0, 805306624)),
        'qwen3-32b down': ((2.9416, 2.9991, 3.0476), (3.0560, 3.0601, 3.0890), (0, 167772416)),
        'llama-3-8b gate_up': ((2.7024, 2.7193, 2.7298), (3.0461, 3.0594, 3.0954), (0, 939524352)),
        'llama-3-70b down': ((5.3525, 5.5330, 5.5804), (5.4171, 5.4566, 5.5361), (0, 268435712)),
        'qwen3-32b gate_up': ((6.0584, 6.1369, 6.3000), (6.7763, 6.7872, 6.8585), (0, 1677721856)),
    },
    Made.INPUT_GRADIENT: {
        'qwen3.5-4b key': ((0.0576, 0.0719, 0.0780), (0.0677, 0.0814, 0.0830), (0, 0)),
        'llama-3-8b o': ((0.3380, 0.3679, 0.4010), (0.3613, 0.3870, 0.4036), (0, 0)),
        'llama-3-8b qkv': ((0.4964, 0.5194, 0.5325), (0.5309, 0.5521, 0.5637), (0, 0)),
        'qwen3.5-4b up': ((0.4838, 0.4844, 0.4933), (0.4928, 0.5087, 0.5129), (0, 0)),
        'qwen3.5-4b down': ((0.4914, 0.4945, 0.5044), (0.5512, 0.5650, 0.5671), (0, 0)),
        'qwen3-32b qkv': ((1.0518, 1.0687, 1.1387), (1.2029, 1.2165, 1.2276), (0, 0)),
        'qwen3-8b down': ((1.1284, 1.1335, 1.1447), (1.3042, 1.3069, 1.3103), (0, 402653440)),
        'llama-3-8b down': ((1.3164, 1.3187, 1.3279), (1.4844, 1.5258, 1.5382), (0, 469762304)),
        'qwen3-8b gate_up': ((2.2155, 2.2996, 2.3307), (2.2958, 2.3161, 2.3538), (0, 134217984)),
        'qwen3-32b down': ((2.9899, 3.0100, 3.0221), (3.3060, 3.3156, 3.3850), (0, 838861056)),
        'llama-3-8b gate_up': ((2.6437, 2.6847, 2.7155), (2.6584, 2.7252, 2.7411), (0, 134217984)),
        'llama-3-70b down': ((5.3087, 5.5183, 5.5419), (5.8066, 5.8347, 5.8973), (0, 939524352)),
        'qwen3-32b gate_up': ((6.2287, 6.2837, 6.3607), (5.9797, 6.1011, 6.3124), (0, 167772416)),
    },
    Made.KERNEL_GRADIENT: {
        'qwen3.5-4b key': ((0.0766, 0.0909, 0.1019), (0.0559, 0.0713, 0.0722), (31459120, 33554688)),
        'llama-3-8b o': ((0.3402, 0.3653, 0.3707), (0.3409, 0.3554, 0.3637), (0, 33554688)),
        'llama-3-8b qkv': ((0.5200, 0.5374, 0.5446), (0.5269, 0.5442, 0.5506), (0, 33554688)),
        'qwen3.5-4b up': ((0.5163, 0.5303, 0.5366), (0.4918, 0.4999, 0.5053), (0, 33554688)),
        'qwen3.5-4b down': ((0.5172, 0.5214, 0.5327), (0.4931, 0.4966, 0.5061), (0, 33554688)),
        'qwen3-32b qkv': ((1.2356, 1.2370, 1.2406), (1.1757, 1.1932, 1.2084), (0, 33554688)),
        'qwen3-8b down': ((1.1298, 1.1368, 1.1494), (1.1296, 1.1522, 1.1623), (0, 33554688)),
        'llama-3-8b down': ((1.3143, 1.3155, 1.3334), (1.3089, 1.3318, 1.3349), (0, 33554688)),
        'qwen3-8b gate_up': ((2.3666, 2.4125, 2.4711), (2.3413, 2.3552, 2.3777), (0, 33554688)),
        'llama-3-8b gate_up': ((2.7368, 2.7468, 2.7925), (2.7369, 2.7533, 2.7629), (0, 33554688)),
        'qwen3-32b down': ((3.0168, 3.0678, 3.0818), (2.9841, 3.0204, 3.0299), (0, 33554688)),
        'llama-3-70b down': ((5.6084, 5.6540, 5.6886), (5.3422, 5.4215, 5.4514), (0, 33554688)),
        'qwen3-32b gate_up': ((6.2557, 6.3101, 6.3473), (6.0850, 6.3026, 6.3305), (0, 33554688)),
    },
})
M_NG_33_G4 = Scoreboard('sm120', 8192, (
    "M-NG-33b-g4 (RTX PRO 6000, aec6662b1 = GEMM 4157, rounds_window 5 rounds, CUPTI, XLA's cuBLASLt "
    "compiled and tuned at run, Mosaic temporaries zero in all 39)"), {
    Made.FORWARD: {
        'qwen3.5-4b key': ((0.1622, 0.1623, 0.1712), (0.1238, 0.1253, 0.1254), (0, 67109120)),
        'llama-3-8b o': ((0.7560, 0.7591, 0.8002), (0.7976, 0.7994, 0.8024), (0, 134217984)),
        'llama-3-8b qkv': ((1.0876, 1.0907, 1.1571), (1.2345, 1.2378, 1.2398), (0, 201326848)),
        'qwen3.5-4b up': ((1.0337, 1.0373, 1.0426), (1.2792, 1.2814, 1.2816), (0, 301990144)),
        'qwen3.5-4b down': ((1.1769, 1.1781, 1.1790), (0.9719, 0.9749, 0.9766), (0, 83886336)),
        'qwen3-32b qkv': ((2.1797, 2.1901, 2.1982), (2.3549, 2.3651, 2.3682), (0, 335544576)),
        'qwen3-8b down': ((2.2458, 2.2531, 2.2608), (2.1401, 2.1462, 2.1500), (0, 134217984)),
        'llama-3-8b down': ((2.6260, 2.6348, 2.6490), (2.4570, 2.4687, 2.4771), (0, 134217984)),
        'qwen3-8b gate_up': ((4.2623, 4.2911, 4.3010), (4.6218, 4.6366, 4.6434), (0, 805306624)),
        'qwen3-32b down': ((5.6739, 5.6994, 5.7462), (5.4109, 5.4420, 5.4565), (0, 167772416)),
        'llama-3-8b gate_up': ((4.7642, 4.7911, 4.8059), (5.4014, 5.4241, 5.4348), (0, 939524352)),
        'llama-3-70b down': ((9.5527, 9.7053, 9.7926), (9.5253, 9.5928, 9.6273), (0, 268435712)),
        'qwen3-32b gate_up': ((10.4970, 10.5961, 10.6213), (11.4706, 11.5115, 11.5324), (0, 1677721856)),
    },
    Made.INPUT_GRADIENT: {
        'qwen3.5-4b key': ((0.1577, 0.1580, 0.1691), (0.1648, 0.1653, 0.1658), (0, 83886336)),
        'llama-3-8b o': ((0.8628, 0.8641, 0.9333), (0.7984, 0.8004, 0.8033), (0, 134217984)),
        'llama-3-8b qkv': ((1.3132, 1.3141, 1.4222), (1.1479, 1.1528, 1.1542), (0, 134217984)),
        'qwen3.5-4b up': ((1.3074, 1.3094, 1.3604), (0.9932, 0.9957, 1.0000), (0, 83886336)),
        'qwen3.5-4b down': ((1.2790, 1.2829, 1.2879), (1.2824, 1.2836, 1.2849), (0, 301990144)),
        'qwen3-32b qkv': ((2.5545, 2.5590, 2.5679), (2.1578, 2.1667, 2.1724), (0, 167772416)),
        'qwen3-8b down': ((2.5311, 2.5356, 2.5383), (2.3866, 2.3999, 2.4047), (0, 402653440)),
        'llama-3-8b down': ((2.6487, 2.6531, 2.6596), (2.7923, 2.8027, 2.8082), (0, 469762304)),
        'qwen3-8b gate_up': ((5.0740, 5.0961, 5.1144), (4.1435, 4.1596, 4.1735), (0, 134217984)),
        'qwen3-32b down': ((6.1353, 6.2071, 6.2227), (6.1342, 6.1646, 6.1800), (0, 838861056)),
        'llama-3-8b gate_up': ((5.9237, 5.9571, 5.9774), (4.8127, 4.8292, 4.8473), (0, 134217984)),
        'llama-3-70b down': ((10.3924, 10.7502, 10.7762), (10.3651, 10.4209, 10.4508), (0, 939524352)),
        'qwen3-32b gate_up': ((12.4375, 12.8176, 13.0801), (10.7801, 10.8181, 10.8754), (0, 167772416)),
    },
    Made.KERNEL_GRADIENT: {
        'qwen3.5-4b key': ((0.1363, 0.1373, 0.1442), (0.1145, 0.1149, 0.1150), (0, 33554688)),
        'llama-3-8b o': ((0.7616, 0.7624, 0.8069), (0.7107, 0.7125, 0.7131), (0, 33554688)),
        'llama-3-8b qkv': ((1.1930, 1.1956, 1.2005), (1.0197, 1.0234, 1.0251), (0, 33554688)),
        'qwen3.5-4b up': ((1.1753, 1.1769, 1.1780), (0.9652, 0.9698, 0.9721), (0, 33554688)),
        'qwen3.5-4b down': ((1.0197, 1.0222, 1.0238), (0.9494, 0.9518, 0.9533), (0, 33554688)),
        'qwen3-32b qkv': ((2.2500, 2.2564, 2.2615), (2.0559, 2.0658, 2.0696), (0, 33554688)),
        'qwen3-8b down': ((2.2310, 2.2372, 2.2433), (2.0245, 2.0298, 2.0337), (0, 33554688)),
        'llama-3-8b down': ((2.4988, 2.5101, 2.5132), (2.2532, 2.2595, 2.2636), (0, 33554688)),
        'qwen3-8b gate_up': ((4.2950, 4.3076, 4.3320), (3.8300, 3.8497, 3.8580), (0, 33554688)),
        'qwen3-32b down': ((5.4806, 5.5106, 5.5276), (5.2102, 5.2303, 5.2488), (0, 33554688)),
        'llama-3-8b gate_up': ((4.9462, 4.9736, 4.9839), (4.4934, 4.5161, 4.5234), (0, 33554688)),
        'llama-3-70b down': ((9.6081, 9.6497, 9.7147), (9.3029, 9.3392, 9.3631), (0, 33554688)),
        'qwen3-32b gate_up': ((10.6364, 10.7223, 10.7636), (10.2803, 10.3363, 10.3568), (0, 33554688)),
    },
})
SCOREBOARDS = (M_NG_45_A100, M_NG_27_GH200, M_NG_33_G4)
"""The receipts timing a generation's dense projections against XLA's dot, each kind XLA's until every
product it timed is at parity, its products at parity the template's (`Scoreboard`): an A100's, a GH200's
and an RTX PRO 6000's at 8,192 tokens (records/scoreboard.md's GEMM rows)."""

STEP_HELD: dict[tuple[str, Made], str] = {
    ('sm90', Made.INPUT_GRADIENT): (
        "tests/step_memory.py (lane/gemm 75fd93d24bd): a trainer step over two and four rematerialized "
        "blocks on two compile-only H100s, its dense input gradients the template's, holds layers' "
        "cotangents past their layer (attention and Kimi delta attention blocks: live growth 63-113 MB past "
        "what the forward saves and the gradients hold), which XLA's dot there does not"),
}
"""Kinds the template's products are at parity in, in time and each product's temporaries (`SCOREBOARDS`),
held on XLA's dot by evidence about the step around them, each with it (kernel-templates.md §1: peak
memory). The GH200's dense input gradient's step leaked through the mapped projections' weight reductions
(row 12898, fixed in `sharding.on_shards`) and an nGPT block's XLA fallback's fp32 product and convert
(`gemm_xla.direct_dtype`). The pinned JAX build (post132) keeps the hold: its `platform_dependent`'s inlined
branch names no checkpointed value (tests/test_miss_driven_rungs.py). Under post135, which does, every step
memory, miss-driven rung and lowering test held with the hold cleared (witness/stepheld-post135-6), so it
goes with the pin that brings that build."""

M_FS_34 = {
    'sm86': ("M-FS-34 / LF-P0-sm86-a10 (an A10, main 45131ca01, XLA tuned on the GPU): the template "
             "1.277x, 1.583x and 1.488x XLA's dot (forward, input and kernel gradient, medians of 13 "
             "projections at 8192 tokens)"),
    'sm89': ("M-FS-34 / LF-P0-sm89-l4 (an L4, main 45131ca01, XLA tuned on the GPU): the template 1.211x, "
             "1.445x and 1.214x XLA's dot (forward, input and kernel gradient, medians of 13 projections at "
             "8192 tokens)")}
"""The generations whose dense products the template runs behind XLA's dot at every kind, by kind alone
(their receipts' cells too noisy to part: spreads to 28%), each with its receipt."""

M_EM_2 = ("M-EM-2 (A100-SXM4-40GB at 1095 MHz, base/kernels 59b96a9b4c2, 400 role-swapped pairs, A/A within "
          "0.993-1.008): the template 1.07x CUTLASS's time, geomean of 5 grouped MoE rows (Qwen3-30B-A3B up "
          "1.193, down 1.090; Mixtral up 0.795, down 1.090; GPT-OSS-20B up 1.244); M-FS-30-a100 (main "
          "45131ca01): Qwen3-30B-A3B down 1.025, up 1.181, GPT-OSS-20B up 1.138")
"""The A100's grouped forward, which CUTLASS's launches hold (row 5623)."""

FOUR_B_TIMED = Timed(frozenset(FOUR_B), 24576, 24576)
"""The rows M-EM-36, M-EM-52, em35d and the B200 grid timed the 4B's projections at (`FOUR_B_CELLS`): 24,576
tokens each, no other count, so the cells hold there alone and a product over other rows takes its kind's
route (kernel-templates.md §1: a cell over its timed rows; row 13167's audit: 24,576 to 49,152 claimed,
32,768 never timed; M-EM-54's 49,152 a whole step's, not each product's)."""

NORMED = ((Timed(frozenset(NORMED_08B), 12288, 12288), M_EM_65, ('sm80', 'sm90')),
          (Timed(frozenset(NORMED_4B_8B), 4096, 12288), M_EM_67, ('sm90',)))
"""The normed projections' receipts: what they timed, the receipt, the generations they ran on, each a
formed product's forward and kernel gradient the template's (M-EM-65, M-EM-67)."""

Cell = tuple[str, Made, bool]
"""A receipt's cells' generation, what they make and whether a prologue forms their rows (`MEASURED`)."""


def timings() -> tuple[Timing, ...]:
    """Every product a receipt timed (`Timing`): the 4B's (`FOUR_B_CELLS`), the normed projections'
    (`NORMED`) and each scoreboard's (`SCOREBOARDS`), the one source `MEASURED` and
    `grouped_matmul.DECLARED` both read."""
    return (*(Timing(generation, made, matrix, FOUR_B_TIMED._replace(stored=STORED[made]), winner, evidence)
              for (generation, made, matrix), (winner, evidence) in FOUR_B_CELLS.items()),
            *(Timing(generation, made, matrix, timed, TEMPLATE, evidence, formed=True)
              for timed, evidence, generations in NORMED for generation in generations
              for made in (Made.FORWARD, Made.KERNEL_GRADIENT) for matrix in sorted(timed.matrices)),
            *(timing for board in SCOREBOARDS for timing in board.timings()))


def _measured() -> dict[Cell, tuple[Timed, ...]]:
    """Each receipt's scope by its cells' generation, what they make and whether a prologue forms their rows
    (`MEASURED`), from `timings`."""
    measured: dict[Cell, tuple[Timed, ...]] = {}
    for timing in timings():
        key = (timing.generation, timing.made, timing.formed)
        measured[key] = (*measured.get(key, ()), timing.timed)
    return measured


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
    """A receipt of products in one launch (`Kind.together`) and the kernel it timed winning: on what
    generation and what they make (`made`), the geometries, rows and arithmetic it timed (`timed`), the kernel
    (`winner`, a grouped selector's kernel by name) and the receipt (`evidence`): the one record a call's cell
    (`grouped_matmul.grouped_case`) and its declared route (`grouped_matmul.DECLARED`) both read (base review
    760be77fe F6)."""

    generation: str
    made: Made
    timed: Timed
    winner: str
    evidence: str


QKV_4B_SHORT = (('chained', 2560, (1024, 1024, 8192)),)
"""Qwen3.5-4B's query, key and value chained input gradient, where M-EM-73 measured the template behind XLA at
1 x 2048 (0.263 vs 0.200 ms, 1.32x; rows 9665, 9768, 10151)."""

TOGETHER = (
    Together('sm90', Made.INPUT_GRADIENT, Timed(frozenset(QKV_4B_SHORT), 2048, 2048), 'apart', M_EM_73),
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

FOUR_B_CELLS: dict[tuple[str, Made, tuple[int, int]], tuple[str, str]] = {
    ('sm90', Made.FORWARD, (2560, 9216)): (TEMPLATE, f"{M_EM_36}: the template 0.866x XLA's dot"),
    ('sm90', Made.FORWARD, (9216, 2560)): (REFERENCE, f"{M_EM_36}: the template 1.01x XLA's dot"),
    ('sm90', Made.FORWARD, (2560, 1024)): (TEMPLATE, f"{M_EM_36}: the template 0.96x XLA's dot"),
    ('sm90', Made.FORWARD, (2560, 64)): (REFERENCE, f"{M_EM_36}: the template 1.294x XLA's dot"),
    ('sm90', Made.INPUT_GRADIENT, (2560, 9216)): (TEMPLATE, f"{M_EM_36}: the template 0.943x XLA's dot"),
    ('sm90', Made.INPUT_GRADIENT, (9216, 2560)): (TEMPLATE, f"{M_EM_36}: the template 0.912x XLA's dot"),
    ('sm90', Made.INPUT_GRADIENT, (2560, 1024)): (TEMPLATE, f"{M_EM_36}: the template 0.949x XLA's dot"),
    ('sm90', Made.INPUT_GRADIENT, (2560, 64)): (TEMPLATE, f"{M_EM_36}: the template 0.93x XLA's dot"),
    ('sm90', Made.KERNEL_GRADIENT, (2560, 9216)): (
        REFERENCE, f"{M_EM_52}: the template 1.042x XLA's dot (1.559 against 1.496 ms)"),
    ('sm90', Made.KERNEL_GRADIENT, (9216, 2560)): (
        TEMPLATE, f"{M_EM_52}: the template 0.755x XLA's dot (1.493 against 1.977 ms)"),
    ('sm90', Made.KERNEL_GRADIENT, (2560, 1024)): (
        TEMPLATE, f"{M_EM_52}: the template 0.784x XLA's dot (0.189 against 0.241 ms)"),
    ('sm90', Made.KERNEL_GRADIENT, (2560, 64)): (
        TEMPLATE, f"{M_EM_52}: the template 0.39x XLA's dot (0.053 against 0.136 ms)"),
    ('sm100', Made.FORWARD, (2560, 64)): (
        REFERENCE, f"{B200_GRID}: the template 1.216x XLA's dot (em35d 1.225x)"),
    ('sm100', Made.INPUT_GRADIENT, (2560, 9216)): (REFERENCE, f"{EM35D}: the template 1.009x XLA's dot"),
    ('sm100', Made.KERNEL_GRADIENT, (2560, 9216)): (
        REFERENCE, f"{B200_GRID}: the template 1.355x XLA's dot (em35d 1.303x)"),
    ('sm100', Made.KERNEL_GRADIENT, (2560, 1024)): (
        REFERENCE, f"{B200_GRID}: the template 1.148x XLA's dot (em35d 1.326x)"),
}
"""The 4B's projections (`FOUR_B`) at 24,576 tokens: (generation, what the product makes, its
matrix) to its winner's route and the receipt, each a cell where its winner is not its kind's route
(`grouped_matmul.DECLARED`)."""


MEASURED = _measured()
"""The dense products each generation's receipts timed (`timings`), by what the product makes and whether a
prologue forms its rows: one of them is a cell of its own (`Kind.matrix` over `Kind.span`) where its winner
is not its kind's route (`grouped_matmul.DECLARED`), its kind's otherwise (`grouped_matmul.GroupedCase.cell`);
every other product its kind's."""
