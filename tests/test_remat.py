"""Rematerialization must be invisible to everything except memory use.

Same parameter tree, same forward values, same gradients: a difference in
any of them would invalidate checkpoints or change what a run converges to.
"""

import contextlib
import io
import os
import shutil
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from flax import linen as nn
from jax.ad_checkpoint import checkpoint_name, print_saved_residuals

from dew.nn.attention import cudnn_runs
from dew.nn.backbones.dit import SimpleDiT
from dew.nn.backbones.mmdit import HierarchicalMMDiT, SimpleMMDiT
from dew.nn.backbones.ssm_dit import HybridSSMAttentionDiT
from dew.nn.backbones.uvit import SimpleUDiT
from dew.nn.backbones.video_dit import VideoDiT
from dew.nn.dit import FUSED_ATTENTION_FORWARD, TextContext, remat_block, saved_through_remat

RES = 32

BUILDERS = {
    'simple_dit': lambda remat: SimpleDiT(
        patch_size=4, emb_features=64, num_layers=2, num_heads=2, mlp_ratio=2, remat=remat),
    'simple_udit': lambda remat: SimpleUDiT(
        patch_size=4, emb_features=64, num_layers=2, num_heads=2, mlp_ratio=2, remat=remat),
    'simple_mmdit': lambda remat: SimpleMMDiT(
        patch_size=4, emb_features=64, num_layers=2, num_heads=2, mlp_ratio=2, remat=remat),
    'hierarchical_mmdit': lambda remat: HierarchicalMMDiT(
        base_patch_size=2, emb_features=(32, 64, 96), num_layers=(1, 1, 1),
        num_heads=(2, 2, 2), mlp_ratio=2, remat=remat),
    'hybrid_dit': lambda remat: HybridSSMAttentionDiT(
        patch_size=4, emb_features=64, num_layers=2, num_heads=2, mlp_ratio=2, remat=remat),
}


def image_inputs(rng):
    return (jax.random.normal(rng, (2, RES, RES, 3)), jnp.ones((2,)),
            TextContext(jnp.ones((2, 77, 768), jnp.float32), jnp.ones((2, 77), bool)))


def saved_residuals(f, *args):
    """The lines jax prints for the values a remat policy keeps in `f`."""
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        print_saved_residuals(f, *args)
    return printed.getvalue().splitlines()


def compare_forward_and_backward(plain, remat, params, *inputs):
    def loss(model, p):
        return jnp.sum(model.apply(p, *inputs) ** 2)

    assert jnp.allclose(loss(plain, params), loss(remat, params), rtol=1e-5, atol=1e-4)
    g_plain = jax.grad(lambda p: loss(plain, p))(params)
    g_remat = jax.grad(lambda p: loss(remat, p))(params)
    for a, b in zip(jax.tree.leaves(g_plain), jax.tree.leaves(g_remat), strict=True):
        assert jnp.allclose(a, b, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize('choice', [True, 'full'])
@pytest.mark.parametrize('arch', sorted(BUILDERS))
def test_remat_keeps_parameter_tree_identical(rng, arch, choice):
    x, temb, ctx = image_inputs(rng)
    plain = BUILDERS[arch](remat=False).init(rng, x, temb, ctx)
    remat = BUILDERS[arch](choice).init(rng, x, temb, ctx)

    def paths(tree):
        return [jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_leaves_with_path(tree)]

    assert paths(plain) == paths(remat), "remat changed the checkpoint layout"


@pytest.mark.parametrize('choice', [True, 'full'])
@pytest.mark.parametrize('arch', sorted(BUILDERS))
def test_remat_preserves_outputs_and_gradients(rng, arch, choice):
    x, temb, ctx = image_inputs(rng)
    plain, remat = BUILDERS[arch](remat=False), BUILDERS[arch](choice)
    compare_forward_and_backward(plain, remat, plain.init(rng, x, temb, ctx), x, temb, ctx)


@pytest.mark.skipif(jax.default_backend() != 'cpu',
                    reason='fp32 remat is bit-exact on the CPU backend, not across GPU fusions')
def test_remat_recompute_is_bit_exact(rng):
    """The recomputed forward runs the same ops on the same saved values, so
    on CPU fp32 nothing is merely close: a single differing bit would mean
    the policy dropped something the backward pass then rebuilt differently."""
    x, temb, ctx = image_inputs(rng)
    plain, remat = BUILDERS['simple_dit'](remat=False), BUILDERS['simple_dit'](remat=True)
    params = plain.init(rng, x, temb, ctx)

    def loss(model, p):
        return jnp.sum(model.apply(p, x, temb, ctx) ** 2)

    assert jnp.allclose(loss(plain, params), loss(remat, params), rtol=0, atol=0)
    g_plain = jax.grad(lambda p: loss(plain, p))(params)
    g_remat = jax.grad(lambda p: loss(remat, p))(params)
    for a, b in zip(jax.tree.leaves(g_plain), jax.tree.leaves(g_remat), strict=True):
        assert jnp.allclose(a, b, rtol=0, atol=0)


@pytest.mark.parametrize('remat', [False, True])
def test_norms_keep_the_residual_stream_in_its_compute_dtype(rng, remat):
    """A norm reduces over the width in fp32, and differentiated as flax
    writes it that upcast is a residual twice over: the fp32 copy of the
    input, and the fp32 normalized activations. A bf16 stream was therefore
    saved at fp32 twice per norm, which on a SimpleDiT step at batch 64 and
    128px was 36 fp32 [64, 1024, 256] tensors, 2.35 GiB, against 48 bf16 ones
    for the stream itself.

    `normalized_in_fp32` recomputes that half. What is left at the stream's
    shape is at most one fp32 tensor for the whole model, the fp32 output
    head's own promoted input, and one bf16 tensor per norm: the norm's input, which is
    the stream as the block already holds it.
    """
    batch, features, layers = 2, 64, 2
    tokens = (RES // 4) ** 2
    model = SimpleDiT(patch_size=4, emb_features=features, num_layers=layers, num_heads=2,
                      mlp_ratio=2, dtype=jnp.bfloat16, remat=remat)
    x, temb = jnp.zeros((batch, RES, RES, 3), jnp.bfloat16), jnp.ones((batch,))
    params = model.init(rng, x, temb, None)
    lines = saved_residuals(
        lambda p: jnp.sum(model.apply(p, x, temb, None).astype(jnp.float32) ** 2), params)

    stream = f'[{batch},{tokens},{features}]'
    in_fp32 = [line for line in lines if f'f32{stream}' in line]
    # At most the output head's promoted input; jax 0.11.2 saves none.
    assert len(in_fp32) <= 1 and all('promote_dtype' in line for line in in_fp32), in_fp32
    assert [line for line in lines if f'bf16{stream}' in line]
    # What a norm leaves behind is its own cast output and nothing else: the
    # fp32 reductions are recomputed rather than named, so a block's remat
    # policy still accounts for every name it keeps (`RESIDUALS`).
    assert all('bf16' in line for line in lines if '(layer_normalized)' in line)
    assert not [line for line in lines if "named 'norm" in line]


class BatchedDotBlock(nn.Module):
    """A batched einsum, the shape attention scores have, optionally carrying
    the name `scaled_dot_product_attention` puts on its output."""

    named: bool

    @nn.compact
    def __call__(self, x):
        scores = jnp.einsum('bhqd,bhkd->bhqk', x, x)
        return jnp.tanh(checkpoint_name(scores, 'attention_output') if self.named else scores)


@pytest.mark.parametrize('named', [True, False])
def test_remat_policy_saves_what_the_attention_name_marks(rng, named):
    """A batched dot is not a dot the policy saves on its own and it is no
    fused kernel either, so whether this value survives the backward pass is
    decided by the name alone."""
    x = jax.random.normal(rng, (2, 2, 8, 4))
    block = remat_block(BatchedDotBlock, enabled=True)(named=named)
    params = block.init(rng, x)
    lines = saved_residuals(lambda arr: jnp.sum(block.apply(params, arr) ** 2), x)

    inside = [line for line in lines if 'BatchedDotBlock' in line]
    assert bool(inside) == named
    assert all("f32[2,2,8,8] named 'attention_output'" in line for line in inside)


def test_remat_policy_keeps_attention_output_and_drops_the_scores(rng):
    """On the reference attention path the scores are a plain batched einsum,
    which is the [B, H, Q, K] residual remat exists to avoid; the kernel's
    [B, S, H, D] output is the cheap thing to keep, once per layer."""
    layers, heads, features = 2, 2, 64
    tokens, head_dim = (RES // 4) ** 2, features // 2
    model = SimpleDiT(patch_size=4, emb_features=features, num_layers=layers,
                      num_heads=heads, mlp_ratio=2, remat=True)
    x, temb, ctx = image_inputs(rng)
    params = model.init(rng, x, temb, ctx)
    lines = saved_residuals(lambda p: jnp.sum(model.apply(p, x, temb, ctx) ** 2), params)

    named = [line for line in lines if "named 'attention_output'" in line]
    assert len(named) == layers
    assert all(f'f32[2,{tokens},{heads},{head_dim}]' in line for line in named)
    assert not [line for line in lines if f'f32[2,{heads},{tokens},{tokens}]' in line]


def test_fused_attention_forward_is_named_as_the_policy_matches_it():
    """The policy recognises the fused forward by the primitive's name, so a
    rename in jax would silently cost a second flash forward per layer rather
    than fail. Read the names off jax's own primitives instead."""
    from jax._src.cudnn.fused_attention_stablehlo import (
        _dot_product_attention_fwd_p,
        _dot_product_attention_fwd_p_wrapper,
    )

    assert {str(_dot_product_attention_fwd_p),
            str(_dot_product_attention_fwd_p_wrapper)} == set(FUSED_ATTENTION_FORWARD)
    assert saved_through_remat(_dot_product_attention_fwd_p_wrapper)
    assert not saved_through_remat(jax.lax.tanh_p)


@pytest.mark.skipif(not cudnn_runs(jnp.zeros((1, 8, 2, 64), jnp.bfloat16)),
                    reason='needs the fused cudnn kernel, which this backend will not run')
def test_fused_attention_runs_once_per_layer_under_remat():
    """What the policy is for: the flash forward stays out of the backward
    pass, so a rematerialized step holds one fused call per layer, not two."""
    layers = 2
    model = SimpleDiT(patch_size=4, emb_features=256, num_layers=layers, num_heads=4,
                      mlp_ratio=4, remat=True, dtype=jnp.bfloat16, attention_impl='auto')
    x, temb = jnp.zeros((2, RES, RES, 3)), jnp.ones((2,))
    params = model.init(jax.random.PRNGKey(0), x, temb, None)
    step = jax.jit(jax.grad(
        lambda p: jnp.sum(model.apply(p, x, temb, None).astype(jnp.float32) ** 2)))
    text = step.lower(params).compile().as_text()

    assert text.count('custom_call_target="__cudnn$fmhaSoftmax"') == layers
    assert text.count('custom_call_target="__cudnn$fmhaSoftmaxBackward"') == layers


def test_video_dit_remat_matches():
    """Remat only rewrites the backward pass, so the gradients are where a
    dropped residual or a mismatched body would show, not the forward."""
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (1, 3, 16, 16, 3))
    temb = jnp.ones((1,))
    ctx = TextContext(jnp.ones((1, 77, 768), jnp.float32), jnp.ones((1, 77), bool))
    plain = VideoDiT(patch_size=4, emb_features=32, num_layers=1, num_heads=2, mlp_ratio=1)
    remat = VideoDiT(patch_size=4, emb_features=32, num_layers=1, num_heads=2, mlp_ratio=1,
                     remat=True)
    compare_forward_and_backward(plain, remat, plain.init(rng, x, temb, ctx), x, temb, ctx)


def test_a_step_that_does_not_fit_recomputes_one_rung_more_until_the_ladder_ends():
    """The trainer's remat ladder, weakest first: a decoder climbs from no
    recomputation through 'minimal' to 'full', a diffusion backbone from
    False through 'dots' to 'full', and neither leaves a policy the ladder
    does not name."""
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.nn.backbones.decoder_block import REMAT_POLICIES
    from dew.objectives.base import ProgramModule
    from dew.training.trainer import recompute_more

    class Holding:
        """An objective that trains one module and has no head to tile."""

        def __init__(self, model):
            self.model = model

        def tile_head(self):
            return None

        def program_key(self):
            return (ProgramModule(self.model, None, trained=True),)

        def substitute(self, modules):
            (self.model,) = modules

    decoder = Holding(CausalTransformer(
        vocab_size=16, emb_features=8, num_layers=1, num_heads=2, mlp_features=16, max_seq_len=8))
    climbed = []
    while recompute_more(decoder):
        climbed.append(decoder.model.remat)
    assert climbed == [REMAT_POLICIES['minimal'], REMAT_POLICIES['full']]

    diffusion = Holding(BUILDERS['simple_dit'](remat=True))
    assert recompute_more(diffusion) and diffusion.model.remat == 'full'
    assert not recompute_more(diffusion)

    custom = Holding(decoder.model.clone(remat='save_qkv_proj'))
    assert not recompute_more(custom) and custom.model.remat == REMAT_POLICIES['save_qkv_proj']


def test_a_distillation_recomputes_more_in_its_student_alone_and_resumes_there():
    """The ladder climbs the modules a step trains (`ProgramModule.trained`):
    a frozen teacher runs no backward pass, so it keeps its own remat, and
    the rung a checkpoint records is the student's, which a fresh objective
    climbs back to."""
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.nn.backbones.decoder_block import REMAT_POLICIES
    from dew.objectives.distillation import DistillationObjective
    from dew.objectives.lm import LMObjective
    from dew.training.trainer import climb_to, recompute_more, recompute_record

    def distillation():
        decoder = CausalTransformer(vocab_size=16, emb_features=8, num_layers=1, num_heads=2, mlp_features=16,
                                    max_seq_len=8)
        return DistillationObjective(LMObjective(decoder, seq_len=4), LMObjective(decoder, seq_len=4))

    objective = distillation()
    while recompute_more(objective):
        pass
    (student, _), (teacher, _) = ((entry.module, entry.trained) for entry in objective.program_key())
    assert student.remat == REMAT_POLICIES['full'] and teacher.remat is None
    assert recompute_record(objective) == 'full'
    resumed = distillation()
    climb_to(resumed, {'remat': recompute_record(objective)})
    assert [entry.module.remat for entry in resumed.program_key()] == [REMAT_POLICIES['full'], None]


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="needs GPU allocator memory statistics")
def test_compilation_counts_memory_kept_alive_outside_its_state():
    """A fresh 2 GiB allocator isolates the accounting regression from
    fragmentation left by earlier GPU tests. Its real held buffers force
    the compiler to fit the remaining memory, then the step runs there.
    """
    if jax.device_count() != 1 or shutil.which("nvidia-smi") is None:
        pytest.skip("needs one NVIDIA GPU with memory reporting")
    rows = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=10).stdout.splitlines()
    if len(rows) != 1:
        pytest.skip("the isolated allocator experiment needs one physical GPU")
    name, total, free = (value.strip() for value in rows[0].split(","))
    if name != jax.devices()[0].device_kind:
        pytest.skip("cannot identify the current GPU's physical memory")
    pool_mib, runtime_mib = 2048, 1024
    if int(free) < pool_mib + runtime_mib:
        pytest.skip("needs 2 GiB for the child pool and 1 GiB for its CUDA runtime")
    root = Path(__file__).resolve().parents[1]
    environment = {**os.environ, "JAX_PLATFORMS": "cuda",
                   "PYTHONPATH": os.pathsep.join((str(root / "src"), str(root / "tests"))),
                   "XLA_PYTHON_CLIENT_ALLOCATOR": "bfc",
                   "XLA_PYTHON_CLIENT_MEM_FRACTION": str(pool_mib / int(total)),
                   "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}
    done = subprocess.run(
        [sys.executable, "-c", "from test_remat import _resident_memory_step; _resident_memory_step()"],
        cwd=root, env=environment, capture_output=True, text=True, timeout=180)
    assert done.returncode == 0, done.stdout + done.stderr


def _resident_memory_step():
    """Compile and run with real external buffers in the bounded child pool."""
    import optax

    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.objectives.lm import LMObjective
    from dew.training import Trainer

    device = jax.devices()[0]
    model = CausalTransformer(vocab_size=65536, emb_features=64, num_layers=1,
                              num_heads=2, mlp_features=128, max_seq_len=256,
                              dtype=jnp.bfloat16)
    objective = LMObjective(model, seq_len=256)
    trainer = Trainer(objective, optax.adam(1e-4), key=jax.random.key(0))
    state, _, _ = trainer.place()
    batch = {"text": jnp.zeros((4, 257), jnp.int32)}
    trainer.compile(state, batch)
    assert objective.head_tile is None, "the unconstrained step must keep the whole logits"

    def additional_bytes():
        assert trainer.executable is not None
        stats = trainer.executable.memory_analysis()
        assert stats is not None
        return stats.output_size_in_bytes - stats.alias_size_in_bytes + stats.temp_size_in_bytes

    memory = device.memory_stats()
    assert memory is not None
    reserve = memory["bytes_limit"] - memory["bytes_in_use"] - additional_bytes() // 2
    assert reserve > 0
    allocate = jax.jit(lambda value, size: jnp.broadcast_to(value, (size,)), static_argnums=1)
    block = 64 * 2**20
    zero = jnp.asarray(0, jnp.uint8)
    held = [allocate(zero, min(block, reserve - start)) for start in range(0, reserve, block)]
    jax.block_until_ready(held)
    step = trainer.compile(state, batch)
    memory = device.memory_stats()
    assert memory is not None
    assert additional_bytes() <= memory["bytes_limit"] - memory["bytes_in_use"]
    _, loss, _, finite, _ = jax.block_until_ready(step(state, batch))
    assert bool(finite), float(loss)
    jax.block_until_ready(held)



@pytest.mark.parametrize('options', [None, {'xla_embed_ir_in_executable': False}],
                         ids=['default', 'step_options'])
def test_a_step_that_does_not_fit_compiles_again_one_rung_up(monkeypatch, options):
    """A step that does not fit tiles the head first, then compiles again
    under 'minimal', which fits, and stops there: a memory-tight step takes
    the tiled head before any block is recomputed.

    Where the device has step compiler options (the Triton GEMM fusions off
    on sm80 and sm89), a step that does not fit is compiled once more under
    XLA's defaults before it climbs (`fitting_default`). So the headroom
    answers by the rung it was compiled at, not by how many compiles came
    before it."""
    import optax

    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.nn.backbones.decoder_block import REMAT_POLICIES
    from dew.objectives.lm import LMObjective
    from dew.training import Trainer, trainer as trainer_module

    compiled = []

    def headroom(executable, devices, held=0):
        rung = (trainer.objective.head_tile is not None,
                trainer_module.recompute_record(trainer.objective))
        compiled.append(rung)
        return 0 if rung[1] == 'minimal' else -1

    monkeypatch.setattr(trainer_module, 'step_headroom', headroom)
    monkeypatch.setattr(trainer_module, 'step_compiler_options', lambda objective, rows, frozen: options)
    model = CausalTransformer(vocab_size=32, emb_features=8, num_layers=1, num_heads=1,
                              mlp_features=16, max_seq_len=8)
    trainer = Trainer(LMObjective(model, seq_len=4), optax.sgd(1e-3), key=jax.random.key(0))
    state, _, _ = trainer.place()
    trainer.compile(state, {'text': jnp.zeros((8, 5), jnp.int32)})
    # Each rung that does not fit is compiled once, or twice with options.
    tries = 1 if options is None else 2
    assert compiled == [(False, None)] * tries + [(True, None)] * tries + [(True, 'minimal')]
    assert trainer.objective.model.remat == REMAT_POLICIES['minimal']


def recording_runs(monkeypatch, fits):
    """A decoder trainer factory whose fit check answers by rung: `fits` maps
    a rung (whether the head is tiled, the remat's record) to whether it
    fits, and every rung it does not name fits. Returns the factory and the
    rungs each trainer compiled."""
    import optax

    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.objectives.lm import LMObjective
    from dew.training import Trainer, trainer as trainer_module

    compiled, current = [], []

    def headroom(executable, devices, held=0):
        trainer = current[-1]
        rung = (trainer.objective.head_tile is not None,
                trainer_module.recompute_record(trainer.objective))
        compiled.append(rung)
        return 0 if fits.get(rung, True) else -1

    monkeypatch.setattr(trainer_module, 'step_headroom', headroom)
    monkeypatch.setattr(trainer_module, 'step_compiler_options', lambda objective, rows, frozen: None)

    def run():
        model = CausalTransformer(vocab_size=32, emb_features=8, num_layers=1, num_heads=1,
                                  mlp_features=16, max_seq_len=8)
        current.append(Trainer(LMObjective(model, seq_len=4), optax.sgd(1e-3), key=jax.random.key(0)))
        state, _, _ = current[-1].place()
        current[-1].compile(state, {'text': jnp.zeros((8, 5), jnp.int32)})
        return current[-1]

    return run, compiled


def test_a_later_run_of_a_step_starts_at_the_rung_an_earlier_run_chose(monkeypatch):
    """Identical runs of a step can read different free memory. On an RTX
    4080 the 99M MoE at 8 x 1024 tiled its head in the process that compiled
    the step, whose growing pool kept its autotuner's scratch, and kept the
    whole logits in one that loaded it from the compilation cache (98.2
    against 78.4 ms a step). The first run's rung is recorded beside the
    compilation cache, for the program and the devices, and a later run
    starts there, as a resumed run starts at its checkpoint's rung."""
    fits = {(False, None): False}
    run, compiled = recording_runs(monkeypatch, fits)
    assert run().objective.head_tile is not None
    assert compiled == [(False, None), (True, None)]
    fits.clear()  # the next process finds room for the whole logits
    compiled.clear()
    assert run().objective.head_tile is not None
    assert compiled == [(True, None)]


def test_a_rung_record_for_another_step_is_refused_by_its_path(monkeypatch):
    """A record holds its key, and one whose key is not the run's is refused
    by its path rather than followed: deleting it lets the run decide again."""
    import json

    from dew.training import rungs

    run, _ = recording_runs(monkeypatch, {})
    run()
    [path] = list(rungs.rung_records().glob("*.json"))
    written = json.loads(path.read_text())
    written["key"]["program"] = "0" * 64
    path.write_text(json.dumps(written))
    with pytest.raises(ValueError, match=f"rung record {path}"):
        run()


def refusing_trainer(monkeypatch, refused, error="RESOURCE_EXHAUSTED: Ran out of memory on HBM, the total "
                     "memory required for HLO temporaries (38.47G) exceeds available HBM (31.24G)."):
    """A decoder trainer whose compiles XLA refuses with `error` at every rung
    `refused` names: (whether the head is tiled, the remat's record), as
    XLA:TPU refuses a program whose temporaries exceed HBM. A program it
    compiles fits. Returns the trainer and the rungs compiled, in order."""
    import optax

    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.objectives.lm import LMObjective
    from dew.training import Trainer, trainer as trainer_module

    attempts = []
    compile_lowered = jax.stages.Lowered.compile

    def compile(self, compiler_options=None):
        rung = (trainer.objective.head_tile is not None,
                trainer_module.recompute_record(trainer.objective))
        attempts.append(rung)
        if rung in refused:
            raise jax.errors.JaxRuntimeError(error)
        return compile_lowered(self, compiler_options)

    monkeypatch.setattr(jax.stages.Lowered, 'compile', compile)
    monkeypatch.setattr(trainer_module, 'step_headroom', lambda executable, devices, held=0: 0)
    monkeypatch.setattr(trainer_module, 'step_compiler_options', lambda objective, rows, frozen: None)
    model = CausalTransformer(vocab_size=32, emb_features=8, num_layers=1, num_heads=1,
                              mlp_features=16, max_seq_len=8)
    trainer = Trainer(LMObjective(model, seq_len=4), optax.sgd(1e-3), key=jax.random.key(0))
    return trainer, attempts


def test_a_step_xla_refuses_for_memory_compiles_again_one_rung_up(monkeypatch):
    """XLA:TPU checks a program's temporaries against HBM as it compiles and
    refuses one that does not fit, so there is no executable to measure: on
    a v6e Qwen3-0.6B at 16 x 1024 tokens died at the bottom rung, 38.47G of
    temporaries for 31.24G of HBM. The refusal is the step not fitting, and
    the ladder climbs: the head tiles, then the remat."""
    import numpy as np

    from dew.nn.backbones.decoder_block import REMAT_POLICIES

    trainer, attempts = refusing_trainer(monkeypatch, {(False, None), (True, None)})
    state, _, _ = trainer.place()
    batch = {'text': jnp.zeros((8, 5), jnp.int32)}
    executable = trainer.compile(state, batch)

    assert attempts == [(False, None), (True, None), (True, 'minimal')]
    assert trainer.objective.model.remat == REMAT_POLICIES['minimal']
    state, loss, *_ = executable(state, batch)
    assert np.isfinite(float(loss))
    assert trainer._rung() == {'head_tile': list(trainer.objective.head_tile), 'remat': 'minimal',
                               'xla_defaults': False}


def test_a_step_xla_refuses_at_every_rung_raises_its_refusal(monkeypatch):
    """Where XLA refuses the last rung too, the run stops with the refusal it
    already gave, with no compile of that rung again: on a large program each
    one costs minutes."""
    rungs = {(tiled, remat) for tiled in (False, True) for remat in (None, 'minimal', 'full')}
    trainer, attempts = refusing_trainer(monkeypatch, rungs)
    state, _, _ = trainer.place()

    with pytest.raises(jax.errors.JaxRuntimeError, match="RESOURCE_EXHAUSTED: Ran out of memory on HBM"):
        trainer.compile(state, {'text': jnp.zeros((8, 5), jnp.int32)})
    assert attempts == [(False, None), (True, None), (True, 'minimal'), (True, 'full')]


def test_a_compile_error_that_is_not_about_memory_is_not_a_rung(monkeypatch):
    """Only XLA's out-of-memory refusal means the step does not fit; any
    other compile error is raised as it is, at the rung it happened on."""
    trainer, attempts = refusing_trainer(monkeypatch, {(False, None)},
                                         error="INVALID_ARGUMENT: an unsupported custom call")
    state, _, _ = trainer.place()

    with pytest.raises(jax.errors.JaxRuntimeError, match="INVALID_ARGUMENT"):
        trainer.compile(state, {'text': jnp.zeros((8, 5), jnp.int32)})
    assert attempts == [(False, None)]
    assert trainer.objective.head_tile is None


@pytest.mark.parametrize('activation', ['swiglu', 'geglu', 'geglu_exact'])
def test_a_gated_product_keeps_only_its_16_bit_inputs_for_the_backward(activation):
    """The product runs in fp32. Differentiated as written it kept five fp32
    copies of the MLP's width, which a scanned stack stores per layer."""
    from dew.nn.moe import gated_product

    product = gated_product(activation)
    gate, up = (jnp.ones((2, 8, 32), jnp.bfloat16) * value for value in (0.5, 1.5))
    lines = saved_residuals(lambda g, u: jnp.sum(product(g, u).astype(jnp.float32)), gate, up)
    assert lines and not [line for line in lines if 'f32[2,8,32]' in line], lines
