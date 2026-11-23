"""Decode's check-1 matrix (tests/kernel_matrix.py): a serving step's decode on par with the reference.

Its cells run the kernel choices `decode` (`attention_kernels.DECODE`: a step over a dense cache's key
lengths, as `scaled_dot_product_attention` takes it) and `paged_decode` (`kv_cache.PAGED`: a step over a paged
pool, as `KVStore.decode` takes it) as a serving step calls them, on the cell's GPU. A dense cell holds the
output and its query, key and value gradients under a cotangent to the reference path's (check 1), a paged
cell its output, each with their float64 terms (`attention_oracle.attention_terms`). What each choice serves
is read from its selector: the decode selector's own test (`attention_kernels.decodes`), and the paged
selector taking a paged kernel (`select(PAGED, case)`, as the layer asks it,
`CausalSelfAttention._decode_attention`).
"""

import contextlib

import flax.linen as nn
import generation_support
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from attention_oracle import attention_terms
from gpu_flow import cell
from kernel_matrix import AXES, head_shapes
from reference_error import assert_as_exact

from dew.nn import attention_kernels
from dew.nn.attention import attention_case, scaled_dot_product_attention
from dew.nn.kernels import generation, selection
from dew.nn.kv_cache import PAGED, KVCache, KVSpec, KVStore, PagedCase

ROWS, CAPACITY, PAGE = 4, 512, 16
LENGTHS = (1, 63, 300, 512)
"""A step's rows, each over its own keys of a cache `CAPACITY` slots a row: one key, part of a page, part of
several, every slot; a paged pool's pages `PAGE` slots each, which both paged kernels read (the Pallas
kernel's pages at most its `BLOCK`, the template's several to a key tile)."""

CHOICES = frozenset({"decode", "paged_decode", "attention_mainloop"})
"""The kernel choices (`generation.CHOICES`) its cells run: a dense cache's step's selector and a paged
pool's, and the template's mainloop where a cell names the template's decode."""
KERNELS = {"decode": frozenset({"decode", "cudnn", "template"}),
           "paged_decode": frozenset({"triton", "template", "tpu"})}
"""The kernels its cells run under test, each named by its choice (`generation.chosen`, the `kernel` axis):
the decode selector's hand kernel, cuDNN's and the attention template's decode; the paged selector's Pallas
kernel and the template's decode over the pages on a GPU, jax's Pallas paged attention on a TPU."""


def _choice(case: dict[str, str]) -> str:
    """The kernel choice `case`'s layout runs (`CHOICES`)."""
    return "decode" if case["layout"] == "dense" else "paged_decode"

CAP = 30.0
SEEDS = 8
HEADS = {"gqa-16-8": (16, 8), "mha-8": (8, 8)}


def _hooks(case: dict[str, str], sinks):
    return {"softcap": CAP if case["hook"] == "softcap" else None,
            "sinks": sinks if case["hook"] == "sinks" else None}


def _precision(case: dict[str, str]):
    """The matmul precision `case` runs at: JAX's default (`None`) on a TPU (where a user's default run
    takes `jax`'s Pallas paged attention, which a v5e refuses under the suite's 'highest', rows 7373/8278),
    the ambient precision elsewhere."""
    return jax.default_matmul_precision(None) if case["generation"] == "v5e" else contextlib.nullcontext()


def supported(case: dict[str, str]) -> bool:
    """Whether its choice's selector serves `case` on its GPU at its kernel (tests/kernel_matrix.py): the
    decode selector takes a dense step (`decodes`), the paged selector a paged one, each by the kernel named
    (`selection.select`), which takes it; one this host does not lower for the GPU (`selection.Unlowerable`:
    cuDNN where it loads no cuDNN library) served by that decision there, its cell native alone."""
    heads, kv_heads, width, dtype = head_shapes(case, HEADS)
    with generation.assumed(case["generation"]), _precision(case):
        if case["layout"] == "dense":
            query = jnp.zeros((ROWS, 1, heads, width), dtype)
            key = jnp.zeros((ROWS, CAPACITY, kv_heads, width), dtype)
            sinks = jnp.zeros((heads,), jnp.float32)
            called = attention_case(query, key, key_value_seq_lengths=jnp.asarray(LENGTHS, jnp.int32),
                                    **_hooks(case, sinks))
            op, called = attention_kernels.DECODE, called if attention_kernels.decodes(called) else None
        else:
            op = PAGED
            called = PagedCase(heads=heads, kv_heads=kv_heads, head_dim=width, value_dim=width, dtype=dtype,
                               pool=dtype, page_size=PAGE, quantized=None, groups=1, head_shards=1,
                               softcap=CAP if case["hook"] == "softcap" else None, arithmetic=False,
                               named="auto", rows=ROWS, queries=1, keys=CAPACITY,
                               sinks=case["hook"] == "sinks")
        if called is None or case["kernel"] not in KERNELS[_choice(case)]:
            return False
        try:
            return selection.select(op, called, case["kernel"]).kernel == case["kernel"]
        except selection.KernelRefused:
            return False
        except selection.Unlowerable as unlowered:
            return unlowered.kernel == case["kernel"]


class _Paged(nn.Module):
    """A paged store of `CAPACITY` slots a row: its keys and values written, then one query a row decoded over
    the first `lengths` slots of its row (`KVStore.decode`) under the layer's `softcap` and `sinks`, through
    the paged kernel the selector gives the step's case (`PAGED`), as a layer selects it."""

    softcap: float | None

    @nn.compact
    def __call__(self, key, value, query, lengths, sinks):
        rows, _, kv_heads, width = key.shape
        store = KVStore.open(self, KVSpec(KVCache(page_size=PAGE), rows, CAPACITY, kv_heads, width,
                                          key.dtype))
        if not self.has_variable("cache", "written"):
            self.variable("cache", "written", lambda: True)
            return None
        store.write(key, value, jnp.broadcast_to(jnp.arange(CAPACITY), (rows, CAPACITY)))
        kernel = store.paged_kernel(query, softcap=self.softcap, arithmetic=False, named="auto",
                                    sinks=sinks is not None)
        assert kernel is not None
        return store.decode(query, lengths, kernel, self.softcap, sinks=sinks)


def _step(case: dict[str, str], *, reference: bool):
    """`case`'s step: a dense one's output and its gradients under `dout`, a paged one's output, through its
    choice's route at its kernel (`generation.chosen`) or the reference path."""
    lengths = jnp.asarray(LENGTHS, jnp.int32)

    def step(*operands):
        if reference:
            return stepped(*operands)
        with generation.chosen(**{_choice(case): case["kernel"]}):
            return stepped(*operands)

    def stepped(query, key, value, dout, sinks):
        hooks = _hooks(case, sinks)
        if case["layout"] == "paged" and not reference:
            paged = _Paged(hooks["softcap"])
            cache = paged.init(jax.random.key(0), key, value, query[:, 0], lengths, hooks["sinks"])["cache"]
            return (paged.apply({"cache": cache}, key, value, query[:, 0], lengths, hooks["sinks"],
                                mutable=["cache"])[0][:, None],)

        def attended(query, key, value):
            return scaled_dot_product_attention(query, key, value, key_value_seq_lengths=lengths,
                                                implementation="reference" if reference else "auto", **hooks)

        if case["layout"] == "paged":
            return (attended(query, key, value),)
        out, pull = jax.vjp(attended, query, key, value)
        return (out, *pull(dout.astype(out.dtype)))

    return step


def _terms(case: dict[str, str]):
    """The step's float64 terms (`attention_terms`): the output's, and a dense step's gradients'."""
    keep = (np.arange(CAPACITY)[None, :] < np.asarray(LENGTHS)[:, None])[:, None, None, :]

    def terms(query, key, value, dout, sinks):
        hooks = _hooks(case, sinks)
        sunk = None if hooks["sinks"] is None else np.asarray(hooks["sinks"], np.float64)
        out, dq, dk, dv, *_ = attention_terms(query, key, value, dout, keep, sinks=sunk,
                                              softcap=hooks["softcap"])
        return (out,) if case["layout"] == "paged" else (out, dq, dk, dv)

    return terms


def _drawn(case: dict[str, str]):
    """A draw of `case`'s operands by seed: unit Gaussians in its dtype, the sinks fp32."""
    heads, kv_heads, width, dtype = head_shapes(case, HEADS)

    def draw(seed: int):
        streams = jax.random.split(jax.random.key(seed), 5)
        query, dout = (jax.random.normal(stream, (ROWS, 1, heads, width), dtype) for stream in streams[:2])
        key, value = (jax.random.normal(stream, (ROWS, CAPACITY, kv_heads, width), dtype)
                      for stream in streams[2:4])
        return query, key, value, dout, jax.random.normal(streams[4], (heads,), jnp.float32)

    return draw


CASES = (
    ('sm90', 'dense', 'mha-8', 'plain', 'float32', '64', 'decode'),
    ('sm80', 'dense', 'gqa-16-8', 'softcap', 'float32', '128', 'template'),
    ('sm90', 'paged', 'gqa-16-8', 'sinks', 'bfloat16', '64', 'triton'),
    ('sm80', 'paged', 'mha-8', 'sinks', 'float16', '64', 'template'),
    ('sm90', 'dense', 'gqa-16-8', 'sinks', 'float16', '128', 'decode'),
    ('sm80', 'dense', 'mha-8', 'plain', 'bfloat16', '64', 'cudnn'),
    ('sm90', 'paged', 'mha-8', 'softcap', 'bfloat16', '128', 'triton'),
    ('sm90', 'dense', 'mha-8', 'plain', 'float16', '128', 'cudnn'),
    ('sm90', 'paged', 'gqa-16-8', 'plain', 'bfloat16', '64', 'template'),
    ('sm80', 'dense', 'gqa-16-8', 'softcap', 'bfloat16', '64', 'decode'),
    ('sm80', 'paged', 'gqa-16-8', 'plain', 'bfloat16', '64', 'triton'),
    ('sm90', 'dense', 'gqa-16-8', 'sinks', 'float32', '64', 'decode'),
    ('sm90', 'dense', 'gqa-16-8', 'softcap', 'float16', '64', 'decode'),
    ('v5e', 'paged', 'gqa-16-8', 'plain', 'bfloat16', '128', 'tpu'),
    ('v5e', 'paged', 'mha-8', 'softcap', 'bfloat16', '128', 'tpu'),
)
"""The cells' cases, each its axis values in `AXES`' order: the cases a ledger row names
(`kernel_matrix.SEEDED`) and every pair of axis values the choices serve (`kernel_matrix.covering("decode")`,
regenerated where the axes or the selectors' support change)."""


@cell("decode", covers=lambda case: dict(zip(AXES["decode"], case, strict=True)))
@pytest.mark.parametrize("case", [pytest.param(case, marks=generation_support.pinned(case[0]))
    for case in CASES], ids="-".join)
def a_decode_step_is_on_par(case: tuple[str, ...]) -> None:
    """A decode step of `case` on its device, through its choice's route at its kernel, on par with the
    reference path's (check 1); its routes its layout's choice at that kernel (`KERNELS`)."""
    named = dict(zip(AXES["decode"], case, strict=True))
    with _precision(named), generation.routes() as routes:
        assert_as_exact(jax.jit(_step(named, reference=False)), jax.jit(_step(named, reference=True)),
                        _drawn(named), "-".join(case), seeds=SEEDS, terms=_terms(named))
    recorded = {(found.choice, found.kernel) for found in routes}
    unnamed = sorted(pair for pair in recorded
                     if pair[0] not in CHOICES and pair[1] not in ("xla", "reference"))
    assert (_choice(named), named["kernel"]) in recorded and not unnamed, (sorted(recorded), unnamed)


def per_shard(choice: str, kernel: str) -> dict:
    """`kernel`'s per-shard cells (`sharded.Shard`, tests/test_per_shard.py): its first case on `choice`'s
    layout (`CASES`), an H100's where it has one, none where it has none, as a decode step through its
    choice's route at `kernel`, over its heads a two-device tensor axis splits as the attention layer maps
    them (the query's, the cache's and a sink's a head), under the default logical axis rules: the output, and
    a dense step's gradients, on par with one device's from the reference path in float64, as the case's
    generation."""
    from flax import linen as nn
    from jax.sharding import PartitionSpec
    from sharded import Shard

    from dew.mesh import DEFAULT_RULES

    layout = "dense" if choice == "decode" else "paged"
    cases = [dict(zip(AXES["decode"], case, strict=True)) for case in CASES
             if case[1] == layout and case[-1] == kernel]
    if not cases:
        return {}
    case = next((case for case in cases if case["generation"] == "sm90"), cases[0])

    def specs(mesh) -> list:
        del mesh
        heads = PartitionSpec(None, None, "tensor")
        return [heads, heads, heads, heads, PartitionSpec("tensor")]

    @contextlib.contextmanager
    def shard_context():
        with nn.logical_axis_rules(DEFAULT_RULES), _precision(case):
            yield

    return {"tensor": Shard(_step(case, reference=False), _drawn(case), specs, _step(case, reference=True),
                            SEEDS, "tensor", context=shard_context, target=case["generation"])}


def assert_paged_decode_on_par(model, label: str, *, seeds: int, steps: int = 8, ref_impl: str = "auto"):
    """Assert a paged `CausalTransformer` decode, `model(layout, dtype)`, is on par with its compact cache's
    on `ref_impl` (check 1, `generation.chosen(attention=...)`), the truth that cache's in float64 on the
    reference path."""
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.nn.kv_cache import KVCache

    def decoded(layout, dtype, implementation="auto"):
        def run(params, tokens):
            decoder = model(layout, dtype)
            found, cache = [], decoder.apply(params, 2, method=CausalTransformer.init_cache,
                                             mutable=["cache"])[1]["cache"]
            for step in range(tokens.shape[1]):
                logits, updated = decoder.apply({**params, "cache": cache},
                                                tokens[:, step:step + 1], decode=True, mutable=["cache"])
                cache = updated["cache"]
                found.append(logits.astype(jnp.promote_types(logits.dtype, jnp.float32)))
            return jnp.stack(found)

        compiled = jax.jit(run)
        if implementation == "auto":
            return compiled

        def on_kernel(params, tokens):
            with generation.chosen(attention=implementation):
                return compiled(params, tokens)
        return on_kernel

    def draw(seed):
        tokens = jax.random.randint(jax.random.key(seed), (2, steps), 0, 64)
        return model(KVCache(), "bfloat16").init(jax.random.key(seed + 1000), tokens), tokens

    assert_as_exact(decoded(KVCache(page_size=16), "bfloat16"),
                    decoded(KVCache(), "bfloat16", ref_impl), draw, label,
                    seeds=seeds, truth_fn=decoded(KVCache(), "float64", "reference"))
