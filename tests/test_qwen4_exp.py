"""Qwen4-Exp (ledger 849): tools/qwen4_exp_reference.py writes qwen4-exp-tiny from transformers' own class,
whose logits and greedy continuation tests/test_hf_reference_decoders.py holds Dew's to; this file holds
what the tiny cannot: the release's n-gram lookups, whose primes it cannot hold, its write-back and the
release's wrapper."""

import json
from pathlib import Path

import jax
import numpy as np
from interop_support import assert_reloads, fixture_ids, token_draw, transformers_arms, transformers_model
from reference_error import as_given, assert_as_exact

from dew.interop import Pretrained, PretrainedDecoder
from dew.interop.hf_decoders import translate_config
from dew.nn.engram import Engram, EngramHashes

ROOT = Path(__file__).parent / 'fixtures' / 'hf'
RELEASE = ROOT / 'qwen3.8-flash-next'
TINY = ROOT / 'qwen4-exp-tiny'


def test_the_releases_ngram_lookups_hash_its_ids_as_transformers_does():
    """Qwen3.8-Flash-Next's n-gram per-layer embedding, read from the release's own config, hashes rows
    with end-of-sequence ids inside them and before them to the ids transformers' `NGramEmbedding`
    gathers at that config, bit for bit (release_ple_ids.npz): SplitMix64's multipliers, the release's
    primes from 20000003 on, which pass 2**24, and each look-back stopped at an end of sequence; its
    multipliers and primes are the buffers transformers builds. A position an image holds hashes its
    placeholder id as any other, transformers' lookup reading the ids alone."""
    config = translate_config(json.loads((RELEASE / 'config.json').read_text())['text_config'])
    spec = Engram(**config['engram'])
    recorded = np.load(RELEASE / 'release_ple_ids.npz')
    tokens = recorded['input_ids']
    hashes = EngramHashes(spec=spec, vocab_size=config['vocab_size'])
    media = np.zeros(tokens.shape, bool)
    media[:, 6:9] = True
    for marked in (None, media):
        ids, _ = hashes.init_with_output(jax.random.key(0), tokens, None, None, decode=False, media=marked)
        np.testing.assert_array_equal(np.asarray(ids)[:, :, 0], recorded['ple_ids'])
    np.testing.assert_array_equal(spec.multipliers[0], recorded['layer_multipliers'])
    np.testing.assert_array_equal(np.ravel(spec.primes[0]), recorded['vocab_sizes'])


EXPORT_SEEDS = 16
"""The export's declared draws of ids against transformers' read of it (check 1), sized by `seeds_for`
paired with Dew's logits on armada."""


def test_an_export_opens_in_transformers_as_the_model_it_was(tmp_path):
    """qwen4-exp-tiny written back (a qwen4_exp_text config, its experts fused, its n-gram table in its row
    shards beside the buffers the class holds) opens in transformers' Qwen4ExpForCausalLM with no tensor
    missing, left over or misshapen, which computes Dew's logits on par from its float64 twin (check 1),
    and Dew reads it back as the model it was."""
    import torch
    from transformers import AutoModelForCausalLM

    loaded = Pretrained.load(TINY, dtype='float32', attention_impl='reference')
    PretrainedDecoder.from_model(loaded.model, loaded.variables).save(tmp_path)
    model, report = AutoModelForCausalLM.from_pretrained(
        tmp_path, dtype=torch.float32, attn_implementation='eager', experts_implementation='eager',
        output_loading_info=True)
    assert not any(report[key] for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys')), report
    reference, truth = transformers_arms(model.eval(), lambda model, ids: model(ids, use_cache=False).logits)
    assert_as_exact(lambda ids: loaded.model.apply(loaded.variables, ids), reference, fixture_ids(TINY),
                    'Qwen4-Exp export', seeds=EXPORT_SEEDS, truth_fn=truth)
    assert_reloads(loaded, tmp_path, np.load(TINY / 'input_ids.npy'))


WRAPPER_SEEDS = 8
"""The wrapper's declared draws of ids against transformers' (check 1), sized as `EXPORT_SEEDS`."""


MTP = {'mtp.fc_embedding.weight': (8, 16), 'mtp.layers.0.self_attn.q_proj.weight': (16, 8)}
"""Two of the root names the release stores its prediction layer under (its
model.safetensors.index.json), which transformers' class leaves out, at shapes of no consequence."""


def with_mtp(directory: Path) -> None:
    """`directory`'s single checkpoint file with the release's root prediction-layer names beside it."""
    from safetensors.numpy import load_file, save_file

    stored = load_file(directory / 'model.safetensors')
    drawn = {name: np.random.default_rng(len(name)).normal(size=shape).astype(np.float32)
             for name, shape in MTP.items()}
    save_file({**stored, **drawn}, directory / 'model.safetensors', metadata={'format': 'pt'})


def test_the_release_wrapper_loads_whole_and_text_only_as_its_language_model(tmp_path):
    """Qwen3.8-Flash-Next's wrapper (Qwen4ExpForConditionalGeneration at the release's config, its text half
    qwen4-exp-tiny's config and its tower one block of Qwen 3.5's, transformers' own draw, the release's
    root prediction-layer names beside it) loads whole, its tower Qwen 3.5's, and computes transformers'
    text logits on par from its float64 twin (check 1); its language model alone (`text_only`) computes
    them bit for bit as the whole does; and the whole writes back and reloads as itself."""
    import torch
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForConditionalGeneration

    release = json.loads((RELEASE / 'config.json').read_text())
    text = {key: value for key, value in json.loads((TINY / 'config.json').read_text()).items()
            if key not in ('architectures', 'transformers_version')}
    vision = {**release['vision_config'], 'depth': 1, 'hidden_size': 32, 'intermediate_size': 64,
              'num_heads': 2, 'out_hidden_size': text['hidden_size'], 'num_position_embeddings': 16}
    special = {'image_token_id': 250, 'video_token_id': 251, 'vision_start_token_id': 252,
               'vision_end_token_id': 253}
    torch.manual_seed(0)
    Qwen4ExpForConditionalGeneration(Qwen4ExpConfig(text_config=text, vision_config=vision, **special)).to(
        torch.float32).save_pretrained(tmp_path)
    reference, truth = transformers_arms(
        transformers_model(tmp_path), lambda model, ids: model(input_ids=ids, use_cache=False).logits)
    with_mtp(tmp_path)
    whole = Pretrained.load(tmp_path, dtype='float32', attention_impl='reference')
    alone = Pretrained.load(tmp_path, dtype='float32', attention_impl='reference', text_only=True)
    draw = token_draw(0x4E, 240, (2, 12))
    assert_as_exact(lambda ids: whole.model.apply(whole.variables, ids), reference, draw,
                    'Qwen4-Exp wrapper logits', seeds=WRAPPER_SEEDS, truth_fn=truth)
    (ids,) = draw(0)
    np.testing.assert_array_equal(alone.model.apply(alone.variables, ids),
                                  whole.model.apply(whole.variables, ids))
    whole.save(tmp_path / 'saved')
    assert_reloads(whole, tmp_path / 'saved', ids)


def test_a_loaded_checkpoint_of_a_sharded_table_writes_back_and_reloads(tmp_path):
    """qwen4-exp-tiny, its n-gram table in four row shards and the release's root prediction-layer names
    beside it, loads as without them and writes back as stored (each shard from its slice of the
    stacked table, the buffers and the prediction layer as they were) to a checkpoint that reloads as
    the model it was."""
    import shutil

    source = Path(shutil.copytree(TINY, tmp_path / 'source'))
    with_mtp(source)
    loaded = Pretrained.load(source, dtype='float32', attention_impl='reference')
    plain = Pretrained.load(TINY, dtype='float32', attention_impl='reference')
    ids = np.load(TINY / 'input_ids.npy')
    np.testing.assert_array_equal(loaded.model.apply(loaded.variables, ids),
                                  plain.model.apply(plain.variables, ids))
    loaded.save(tmp_path / 'saved')
    assert_reloads(loaded, tmp_path / 'saved', ids)


NATIVE = ROOT / 'qwen4-exp-native-tiny'

MEDIA_SEEDS = 8
"""The media's declared draws (check 1), sized by `seeds_for` paired with Dew's logits on armada."""


def test_images_and_a_video_read_through_the_processor_as_transformers_reads_them():
    """qwen4-exp-native-tiny (tools/qwen4_exp_reference.py --native) through its own Qwen3-VL processor:
    a prompt over a drawn image and one over a drawn video of two frames, their tower Qwen 3.5's, their
    positions M-RoPE's over the media's grids, and the n-gram lookups hashing the placeholders the
    media fill, compute the logits transformers' Qwen4ExpForConditionalGeneration computes through the
    same processor, from its float64 twin (check 1)."""
    import torch
    import transformers

    loaded = Pretrained.load(NATIVE, dtype='float32', attention_impl='reference')
    assert loaded.processor is not None
    prompts = json.loads((NATIVE / 'prompts.json').read_text())
    processor = transformers.AutoProcessor.from_pretrained(NATIVE, local_files_only=True)
    metadata = [prompts['video_metadata']]

    def dew(image, video):
        pictured = loaded.processor([prompts['image']], images=[[np.asarray(image)]])
        clip = loaded.processor([prompts['video']], videos=[np.asarray(video)], video_metadata=metadata)
        return {name: loaded.model.apply(loaded.variables, inputs.tokens, **inputs.kwargs())
                for name, inputs in (('image', pictured), ('video', clip))}

    def run(network, image, video):
        dtype = next(network.parameters()).dtype
        encoded = {'image': processor(text=[prompts['image']], images=[[image.numpy().astype(np.uint8)]],
                                      return_tensors='pt'),
                   'video': processor(text=[prompts['video']], videos=[video.numpy().astype(np.uint8)],
                                      video_metadata=metadata, return_tensors='pt')}
        return {name: network(**{key: value.to(dtype) if value.is_floating_point() else value
                                 for key, value in inputs.items()}, use_cache=False).logits
                for name, inputs in encoded.items()}

    model = transformers.AutoModelForImageTextToText.from_pretrained(
        NATIVE, dtype=torch.float32, attn_implementation='eager', experts_implementation='eager').eval()
    reference, truth = transformers_arms(model, run)

    def draw(seed):
        rng = np.random.default_rng([0x4E5, seed])
        return (rng.integers(0, 256, (32, 32, 3), dtype=np.uint8),
                rng.integers(0, 256, (2, 32, 32, 3), dtype=np.uint8))

    assert_as_exact(as_given(dew), reference, draw, 'Qwen4-Exp media logits', seeds=MEDIA_SEEDS,
                    truth_fn=truth)
