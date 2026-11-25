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

TEMPLATE, REFERENCE = 'mosaic', 'xla'
"""The routes a scoreboard's cell goes to (`Scoreboard.winner`), by the grouped selector's kernels' names."""


class Scoreboard(NamedTuple):
    """One receipt's dense products on one GPU `generation` at `rows` tokens, the template against XLA's dot
    (cuBLAS/cuBLASLt, compiled and autotuned on the GPU), each product's routes' rounds (`times`: by what it
    made, each projection's template's then XLA's `Ranged`), and the receipt (`evidence`). A product holds the
    template where its rounds lie at or under XLA's, overlapping them or wholly under (`at_parity`), a cell of
    its own over the receipt's rows (`Kind.span`); a kind takes the template whole only once every product
    it timed does, the template matching or beating the best kernel at every shape it serves
    (kernel-templates.md §1; bible §6, 2026-11-24: "templates should replace ref routes on parity"), and keeps
    XLA's dot otherwise, its lagging products its derivation's to climb (`winner`)."""

    generation: str
    rows: int
    evidence: str
    times: Mapping[Made, Mapping[str, tuple[Ranged, Ranged]]]

    def geomean(self, made: Made) -> float:
        """The geometric mean of the template's median over XLA's at each product making `made`."""
        ratios = [template[1] / reference[1] for template, reference in self.times[made].values()]
        return math.exp(sum(map(math.log, ratios)) / len(ratios))

    def at_parity(self, made: Made, projection: str) -> bool:
        """Whether the template's rounds at `projection` making `made` lie at or under XLA's: none of them
        past every one of XLA's."""
        template, reference = self.times[made][projection]
        return template[0] <= reference[2]

    def winner(self, made: Made) -> str:
        """The route a kind making `made` takes: `TEMPLATE` where every product it timed is at parity
        (`at_parity`), else `REFERENCE`."""
        return TEMPLATE if all(self.at_parity(made, name) for name in self.times[made]) else REFERENCE

    def why(self, made: Made, projection: str | None = None) -> str:
        """The receipt's reading for `projection`'s cell, or for its kind's where None."""
        if projection is None:
            behind = sum(not self.at_parity(made, name) for name in self.times[made])
            return (f"{self.evidence}: the template behind XLA's dot at {behind} of {len(self.times[made])} "
                    f"projections at {self.rows} tokens (geomean {self.geomean(made):.3f}x)")
        template, reference = self.times[made][projection]
        apart = 'every round faster' if template[2] < reference[0] else 'rounds overlapping'
        return (f"{self.evidence}: {projection} the template {template[1] / reference[1]:.3f}x XLA's dot "
                f"({template[1]:.4f} against {reference[1]:.4f} ms, {apart})")

    def timed(self, made: Made) -> Timed:
        """The products it timed making `made`, as a cell reads them (`MEASURED`)."""
        return Timed(frozenset(PROJECTIONS[name] for name in self.times[made]), self.rows, self.rows)


M_NG_45_A100 = Scoreboard('sm80', 8192, (
    "M-NG-45-a100 (A100-SXM4-40GB at 1095 MHz, train 1204 b71af0d11, rounds_window 5 rounds, CUPTI, XLA "
    "compiled and autotuned on the GPU)"), {
    Made.FORWARD: {
        'qwen3.5-4b key': ((0.2224, 0.2227, 0.2229), (0.2147, 0.2147, 0.2149)),
        'llama-3-8b o': ((1.2649, 1.2651, 1.2658), (1.2252, 1.2253, 1.2254)),
        'llama-3-8b qkv': ((1.9145, 1.9147, 1.9154), (1.9482, 1.9818, 1.9879)),
        'qwen3.5-4b up': ((1.8148, 1.8154, 1.8163), (1.8159, 1.8175, 1.8191)),
        'qwen3.5-4b down': ((1.7753, 1.7755, 1.7759), (1.6908, 1.6911, 1.6913)),
        'qwen3-32b qkv': ((3.9536, 3.9547, 3.9553), (3.8332, 3.8374, 3.8406)),
        'qwen3-8b down': ((3.7208, 3.7217, 3.7225), (3.6114, 3.6156, 3.6168)),
        'llama-3-8b down': ((4.3310, 4.3314, 4.3333), (4.2492, 4.2813, 4.2859)),
        'qwen3-8b gate_up': ((7.4923, 7.4951, 7.4963), (8.2202, 8.2209, 8.2237)),
        'qwen3-32b down': ((9.6919, 9.6925, 9.6940), (9.2246, 9.2283, 9.2345)),
        'llama-3-8b gate_up': ((8.7250, 8.7284, 8.7296), (9.5830, 9.5874, 9.5886)),
        'llama-3-70b down': ((17.1798, 17.1941, 17.2059), (16.3624, 16.3626, 16.3764)),
        'qwen3-32b gate_up': ((19.4758, 19.4830, 19.4869), (20.7995, 20.8091, 20.8232)),
    },
    Made.INPUT_GRADIENT: {
        'qwen3.5-4b key': ((0.2340, 0.2344, 0.2349), (0.2239, 0.2243, 0.2245)),
        'llama-3-8b o': ((1.3573, 1.3580, 1.3586), (1.2392, 1.2398, 1.2400)),
        'llama-3-8b qkv': ((2.0158, 2.0167, 2.0170), (1.8564, 1.8593, 1.8665)),
        'qwen3.5-4b up': ((1.8075, 1.8077, 1.8104), (1.7154, 1.7156, 1.7219)),
        'qwen3.5-4b down': ((1.9071, 1.9080, 1.9088), (1.8739, 1.8747, 1.8753)),
        'qwen3-32b qkv': ((4.0035, 4.0097, 4.0104), (3.8142, 3.8252, 3.8264)),
        'qwen3-8b down': ((3.9205, 3.9212, 3.9216), (3.8102, 3.8180, 3.8207)),
        'llama-3-8b down': ((4.5875, 4.5890, 4.5896), (4.4804, 4.4823, 4.4863)),
        'qwen3-8b gate_up': ((7.9127, 7.9198, 7.9216), (7.3331, 7.3425, 7.3478)),
        'qwen3-32b down': ((10.1041, 10.1043, 10.1090), (10.5273, 10.5300, 10.5359)),
        'llama-3-8b gate_up': ((9.2100, 9.2122, 9.2124), (8.7936, 8.7952, 8.7967)),
        'llama-3-70b down': ((17.8228, 17.8280, 17.8310), (17.8540, 17.8571, 17.8617)),
        'qwen3-32b gate_up': ((29.9618, 30.2808, 35.7339), (19.0651, 19.0673, 19.0764)),
    },
    Made.KERNEL_GRADIENT: {
        'qwen3.5-4b key': ((0.3050, 0.3053, 0.3053), (0.2909, 0.2911, 0.2917)),
        'llama-3-8b o': ((1.4180, 1.4183, 1.4186), (1.3684, 1.3693, 1.3699)),
        'llama-3-8b qkv': ((2.1240, 2.1243, 2.1252), (1.9759, 1.9765, 1.9771)),
        'qwen3.5-4b up': ((2.0046, 2.0050, 2.0051), (1.8309, 1.8322, 1.8326)),
        'qwen3.5-4b down': ((1.9968, 1.9973, 1.9980), (1.9903, 1.9907, 1.9914)),
        'qwen3-32b qkv': ((4.2638, 4.2648, 4.2650), (3.8879, 3.8889, 3.8894)),
        'qwen3-8b down': ((4.0862, 4.0867, 4.0881), (3.9270, 3.9270, 3.9285)),
        'llama-3-8b down': ((4.7754, 4.7768, 4.7796), (4.5889, 4.5892, 4.5904)),
        'qwen3-8b gate_up': ((8.0055, 8.0066, 8.0076), (7.2275, 7.2311, 7.2317)),
        'llama-3-8b gate_up': ((9.3850, 9.3855, 9.3877), (8.4450, 8.4454, 8.4459)),
        'qwen3-32b down': ((10.5439, 10.5595, 10.5661), (9.9708, 9.9752, 9.9781)),
        'llama-3-70b down': ((18.6177, 18.6187, 18.6195), (17.2770, 17.2828, 17.2841)),
        'qwen3-32b gate_up': ((20.9652, 20.9672, 20.9758), (18.6990, 18.7110, 18.7178)),
    },
})
M_NG_45B_GH200 = Scoreboard('sm90', 8192, (
    "M-NG-45b-gh200 (GH200, train 1204 b71af0d11, rounds_window 5 rounds, CUPTI, XLA compiled and "
    "autotuned on the GPU)"), {
    Made.FORWARD: {
        'qwen3.5-4b key': ((0.0541, 0.0579, 0.0734), (0.0579, 0.0581, 0.0608)),
        'llama-3-8b o': ((0.3838, 0.3945, 0.4100), (0.3565, 0.3602, 0.3867)),
        'llama-3-8b qkv': ((0.5557, 0.5692, 0.5742), (0.5321, 0.5459, 0.5613)),
        'qwen3.5-4b up': ((0.5294, 0.5383, 0.5441), (0.5278, 0.5361, 0.5511)),
        'qwen3.5-4b down': ((0.5302, 0.5371, 0.5471), (0.4959, 0.4999, 0.5166)),
        'qwen3-32b qkv': ((1.1224, 1.1304, 1.1618), (1.1465, 1.1633, 1.1920)),
        'qwen3-8b down': ((1.1062, 1.1658, 1.1811), (1.0727, 1.1455, 1.1641)),
        'llama-3-8b down': ((1.3835, 1.3975, 1.4141), (1.2726, 1.2902, 1.2980)),
        'qwen3-8b gate_up': ((2.3557, 2.3892, 2.4241), (2.6161, 2.6389, 2.6782)),
        'qwen3-32b down': ((2.8841, 3.1104, 3.1535), (3.0293, 3.0420, 3.0656)),
        'llama-3-8b gate_up': ((2.8951, 2.9470, 2.9910), (3.0416, 3.0977, 3.1475)),
        'llama-3-70b down': ((5.6606, 5.8413, 5.8986), (5.4306, 5.5146, 5.6370)),
        'qwen3-32b gate_up': ((6.1114, 6.1863, 6.2434), (6.8085, 7.0086, 7.0711)),
    },
    Made.INPUT_GRADIENT: {
        'qwen3.5-4b key': ((0.0575, 0.0617, 0.0780), (0.0658, 0.0668, 0.0701)),
        'llama-3-8b o': ((0.3812, 0.3925, 0.3939), (0.3584, 0.3659, 0.3923)),
        'llama-3-8b qkv': ((0.5539, 0.5616, 0.5656), (0.5183, 0.5271, 0.5460)),
        'qwen3.5-4b up': ((0.5319, 0.5373, 0.5381), (0.4996, 0.5047, 0.5249)),
        'qwen3.5-4b down': ((0.5290, 0.5353, 0.5434), (0.5455, 0.5530, 0.5606)),
        'qwen3-32b qkv': ((1.1155, 1.1555, 1.2170), (1.1305, 1.1799, 1.2076)),
        'qwen3-8b down': ((1.1919, 1.2093, 1.2228), (1.1418, 1.1799, 1.1897)),
        'llama-3-8b down': ((1.4134, 1.4388, 1.4696), (1.4948, 1.5140, 1.5460)),
        'qwen3-8b gate_up': ((2.4718, 2.4882, 2.5144), (2.2717, 2.3011, 2.3650)),
        'qwen3-32b down': ((3.2106, 3.2707, 3.2921), (3.2420, 3.2904, 3.3169)),
        'llama-3-8b gate_up': ((2.8533, 2.9253, 2.9868), (2.6608, 2.6707, 2.7120)),
        'llama-3-70b down': ((5.8238, 5.8706, 5.9069), (5.5826, 5.6605, 5.7863)),
        'qwen3-32b gate_up': ((6.2228, 6.2728, 6.3411), (6.0851, 6.2065, 6.2997)),
    },
    Made.KERNEL_GRADIENT: {
        'qwen3.5-4b key': ((0.0761, 0.0811, 0.1014), (0.0816, 0.0819, 0.0864)),
        'llama-3-8b o': ((0.3394, 0.3583, 0.3645), (0.3963, 0.4084, 0.4225)),
        'llama-3-8b qkv': ((0.5811, 0.5896, 0.5952), (0.6230, 0.6380, 0.6439)),
        'qwen3.5-4b up': ((0.5355, 0.5406, 0.5484), (0.5274, 0.5753, 0.5891)),
        'qwen3.5-4b down': ((0.5320, 0.5372, 0.5445), (0.6096, 0.6376, 0.6460)),
        'qwen3-32b qkv': ((1.2798, 1.3071, 1.3289), (1.1716, 1.1793, 1.2350)),
        'qwen3-8b down': ((1.2132, 1.2178, 1.2356), (1.1982, 1.2121, 1.2314)),
        'llama-3-8b down': ((1.4015, 1.4154, 1.4291), (1.3761, 1.3863, 1.4072)),
        'qwen3-8b gate_up': ((2.5285, 2.5469, 2.6133), (2.3900, 2.4237, 2.4337)),
        'llama-3-8b gate_up': ((2.9065, 2.9154, 2.9232), (2.7407, 2.7478, 2.7815)),
        'qwen3-32b down': ((3.2576, 3.2687, 3.3220), (3.1517, 3.1740, 3.1834)),
        'llama-3-70b down': ((5.9851, 6.0180, 6.1083), (5.6870, 5.6904, 5.7491)),
        'qwen3-32b gate_up': ((6.6550, 6.7000, 6.7981), (6.0963, 6.1281, 6.2189)),
    },
})
M_FS_51_G4 = Scoreboard('sm120', 8192, (
    "M-FS-51-g4 (RTX PRO 6000, 356fea58c2a's square tiles, on main, rounds_window 5 rounds, CUPTI, XLA's "
    "cuBLASLt compiled and tuned at run)"), {
    Made.FORWARD: {
        'qwen3.5-4b key': ((0.1598, 0.1598, 0.1603), (0.1256, 0.1262, 0.1270)),
        'llama-3-8b o': ((0.7566, 0.7592, 0.7617), (0.8349, 0.8476, 0.8500)),
        'llama-3-8b qkv': ((1.1664, 1.1810, 1.1862), (1.3079, 1.3119, 1.3190)),
        'qwen3.5-4b up': ((1.0574, 1.0630, 1.0700), (1.1808, 1.2108, 1.2144)),
        'qwen3.5-4b down': ((1.1765, 1.1792, 1.1871), (1.0408, 1.0500, 1.0618)),
        'qwen3-32b qkv': ((2.3044, 2.3186, 2.3235), (2.6022, 2.6114, 2.6208)),
        'qwen3-8b down': ((2.3088, 2.3154, 2.3425), (2.1774, 2.2058, 2.2205)),
        'llama-3-8b down': ((2.6780, 2.6846, 2.7008), (2.5559, 2.5889, 2.6080)),
        'qwen3-8b gate_up': ((4.1425, 4.1615, 4.2775), (4.7123, 4.7312, 4.7905)),
        'qwen3-32b down': ((5.6202, 5.6249, 5.6305), (5.1046, 5.1399, 5.1432)),
        'llama-3-8b gate_up': ((4.8387, 4.8417, 4.8443), (5.5109, 5.5123, 5.6060)),
        'llama-3-70b down': ((9.7028, 9.7057, 9.7438), (9.0756, 9.1148, 9.1195)),
        'qwen3-32b gate_up': ((10.6567, 10.6624, 10.7137), (12.4728, 12.4901, 12.5182)),
    },
    Made.KERNEL_GRADIENT: {
        'qwen3.5-4b key': ((0.1379, 0.1381, 0.1385), (0.1372, 0.1382, 0.1394)),
        'llama-3-8b o': ((0.8020, 0.8087, 0.8100), (0.8093, 0.8191, 0.8210)),
        'llama-3-8b qkv': ((1.3182, 1.3197, 1.3273), (1.1348, 1.1410, 1.1556)),
        'qwen3.5-4b up': ((1.0429, 1.0438, 1.0471), (1.0527, 1.0759, 1.0819)),
        'qwen3.5-4b down': ((1.0850, 1.0922, 1.0987), (1.1688, 1.1759, 1.1877)),
        'qwen3-32b qkv': ((2.3736, 2.3908, 2.4077), (2.2339, 2.2582, 2.2654)),
        'qwen3-8b down': ((2.3772, 2.3872, 2.3993), (2.3220, 2.3577, 2.3622)),
        'llama-3-8b down': ((2.5891, 2.6064, 2.6230), (2.5924, 2.5932, 2.5972)),
        'qwen3-8b gate_up': ((4.3729, 4.3765, 4.3843), (4.0955, 4.1114, 4.1161)),
        'llama-3-8b gate_up': ((5.1456, 5.1509, 5.1551), (4.6329, 4.6591, 4.6659)),
        'qwen3-32b down': ((5.6801, 5.7087, 5.7119), (5.5771, 5.5909, 5.5931)),
        'llama-3-70b down': ((9.9946, 10.0428, 10.1028), (9.7426, 9.7645, 9.7718)),
        'qwen3-32b gate_up': ((11.0623, 11.1264, 11.1412), (10.0577, 10.0852, 10.1184)),
    },
})
M_NG_1_G4 = Scoreboard('sm120', 8192, (
    "M-NG-1-g4 (RTX PRO 6000, main 8dc12dd528, rounds_window 5 rounds, CUPTI, XLA's cuBLASLt compiled and "
    "tuned at run)"), {
    Made.INPUT_GRADIENT: {
        'qwen3.5-4b key': ((0.1532, 0.1546, 0.1558), (0.1401, 0.1413, 0.1432)),
        'llama-3-8b o': ((0.8396, 0.8512, 0.8606), (0.8075, 0.8275, 0.8344)),
        'llama-3-8b qkv': ((1.3186, 1.3221, 1.3315), (1.1584, 1.1691, 1.1854)),
        'qwen3.5-4b up': ((1.3030, 1.3126, 1.3138), (1.0062, 1.0205, 1.0344)),
        'qwen3.5-4b down': ((1.2616, 1.2622, 1.2699), (1.1923, 1.2274, 1.2404)),
        'qwen3-32b qkv': ((2.5483, 2.5627, 2.5639), (2.3698, 2.4105, 2.4179)),
        'qwen3-8b down': ((2.5406, 2.5424, 2.5505), (2.4796, 2.4899, 2.5320)),
        'llama-3-8b down': ((2.6632, 2.6856, 2.7068), (2.9094, 2.9573, 2.9639)),
        'qwen3-8b gate_up': ((5.1635, 5.2079, 5.2578), (4.3376, 4.3703, 4.4092)),
        'qwen3-32b down': ((6.3138, 6.3413, 6.3883), (6.0680, 6.0972, 6.1092)),
        'llama-3-8b gate_up': ((6.1589, 6.2052, 6.2498), (5.0662, 5.1166, 5.1246)),
        'llama-3-70b down': ((10.9171, 10.9989, 11.0163), (10.1329, 10.1870, 10.2654)),
        'qwen3-32b gate_up': ((13.1663, 13.2025, 13.2972), (11.4007, 11.4654, 11.5218)),
    },
})
SCOREBOARDS = (M_NG_45_A100, M_NG_45B_GH200, M_FS_51_G4, M_NG_1_G4)
"""The receipts timing a generation's dense projections against XLA's dot, each kind XLA's until every
product it timed is at parity, its products at parity the template's (`Scoreboard`): an A100's, a GH200's
and an RTX PRO 6000's at 8,192 tokens (records/scoreboard.md's GEMM rows)."""

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

Cell = tuple[str, Made, bool]
"""A receipt's cells' generation, what they make and whether a prologue forms their rows (`MEASURED`)."""


def _measured(*timings: tuple[Cell, Timed]) -> dict[Cell, tuple[Timed, ...]]:
    """`timings`' receipts (`Timed`) by their cells' generation, what they make and whether a prologue forms
    their rows, in their order (`MEASURED`)."""
    measured: dict[Cell, tuple[Timed, ...]] = {}
    for key, timed in timings:
        measured[key] = (*measured.get(key, ()), timed)
    return measured


_DENSE_MADE = (Made.FORWARD, Made.INPUT_GRADIENT, Made.KERNEL_GRADIENT)
_FORMED_MADE = (Made.FORWARD, Made.KERNEL_GRADIENT)

MEASURED = _measured(
    *(((generation, made, False), Timed(frozenset(FOUR_B), 24576, 49152))
      for generation in ('sm90', 'sm100') for made in _DENSE_MADE),
    *(((generation, made, True), Timed(frozenset(NORMED_08B), 12288, 12288))
      for generation in ('sm80', 'sm90') for made in _FORMED_MADE),
    *((('sm90', made, True), Timed(frozenset(NORMED_4B_8B), 4096, 12288)) for made in _FORMED_MADE),
    *(((board.generation, made, False), board.timed(made)) for board in SCOREBOARDS for made in board.times))
"""The dense products each generation's receipts timed, by what the product makes and whether a prologue
forms its rows: one of them is a cell of its own (`Kind.matrix` over `Kind.span`), which its receipt's winner
holds, each receipt its own matrices and rows; every other product its kind's. The 4B's (`FOUR_B`) at 24,576
tokens (M-EM-36, M-EM-52, em35d; the 4B step's 49,152, four rows, held the same cells at parity with
cuBLASLt, M-EM-54: 3,418.7 against 3,416.9 ms); the normed projections' (M-EM-65, M-EM-67); the scoreboards'
(`SCOREBOARDS`) at 8,192."""

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
    generation and what they make (`made`), their operands' `dtype`, the geometries and rows it timed
    (`timed`), the kernel (`winner`, a grouped selector's kernel by name) and the receipt (`evidence`): the
    one record a call's cell (`grouped_matmul.grouped_case`) and its declared route
    (`grouped_matmul.DECLARED`) both read (base review 760be77fe F6)."""

    generation: str
    made: Made
    dtype: str
    timed: Timed
    winner: str
    evidence: str


QKV_4B_SHORT = (('chained', 2560, (1024, 1024, 8192)),)
"""Qwen3.5-4B's query, key and value chained input gradient, where M-EM-73 measured the template behind XLA at
1 x 2048 (0.263 vs 0.200 ms, 1.32x; rows 9665, 9768, 10151)."""

TOGETHER = (
    Together('sm90', Made.INPUT_GRADIENT, 'bfloat16',
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
"""The 4B's projections (`FOUR_B`) at 24,576 to 49,152 tokens: (generation, what the product makes, its
matrix) to its winner's route and the receipt, each a cell where its winner is not its kind's route
(`grouped_matmul.DECLARED`)."""
