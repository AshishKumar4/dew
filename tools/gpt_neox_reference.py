"""Write a tiny GPT-NeoX reference with head-interleaved fused qkv tensors."""

from hf_reference import write_device_tiny
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM


def main():
    config = GPTNeoXConfig(vocab_size=128, hidden_size=32, num_hidden_layers=2,
                          num_attention_heads=4, intermediate_size=48,
                          max_position_embeddings=64, hidden_dropout=0, attention_dropout=0,
                          rope_parameters={'rope_type': 'default', 'rope_theta': 10000.,
                                           'partial_rotary_factor': .5},
                          use_parallel_residual=True, bos_token_id=1, eos_token_id=None, pad_token_id=0)
    config._attn_implementation = 'eager'
    write_device_tiny('gpt-neox-tiny', GPTNeoXForCausalLM(config), 'cuda',
                      'PYTHONPATH=src python tools/gpt_neox_reference.py')


if __name__ == '__main__':
    main()
